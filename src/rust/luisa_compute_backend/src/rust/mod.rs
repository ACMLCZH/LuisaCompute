// A Rust implementation of LuisaCompute backend.

use std::{cell::RefCell, sync::Arc};

use crate::SwapChainForCpuContext;

use self::{
    accel::{AccelImpl, MeshImpl},
    resource::{BindlessArrayImpl, BufferImpl},
    stream::{convert_capture, StreamImpl},
    texture::TextureImpl,
};
use super::Backend;
use api::{AccelOption, CreatedBufferInfo, CreatedResourceInfo, PixelStorage};
use libc::c_void;
use log::info;
use luisa_compute_api_types as api;
use luisa_compute_cpu_kernel_defs as defs;
use luisa_compute_ir::{
    codegen::CodeGen,
    context::type_hash,
    ir::{self, Type},
    CArc,
};
use parking_lot::RwLock;

mod accel;
mod resource;
mod shader;
mod stream;
mod texture;
pub struct RustBackend {
    shared_pool: Arc<rayon::ThreadPool>,
    swapchain_context: RwLock<Option<SwapChainForCpuContext>>,
}

impl Backend for RustBackend {
    unsafe fn set_swapchain_contex(&self, ctx: SwapChainForCpuContext) {
        let mut self_ctx = self.swapchain_context.write();
        if self_ctx.is_some() {
            panic!("swapchain context already set");
        }
        *self_ctx = Some(ctx);
    }
    fn query(&self, property: &str) -> Option<String> {
        match property {
            "device_name" => Some("cpu".to_string()),
            _ => None,
        }
    }
    fn create_buffer(
        &self,
        ty: &CArc<ir::Type>,
        count: usize,
    ) -> super::Result<luisa_compute_api_types::CreatedBufferInfo> {
        let size_bytes = ty.size() * count;
        let buffer = Box::new(BufferImpl::new(size_bytes, ty.alignment(), type_hash(&ty)));
        let data = buffer.data;
        let ptr = Box::into_raw(buffer);
        Ok(CreatedBufferInfo {
            resource: CreatedResourceInfo {
                handle: ptr as u64,
                native_handle: data as *mut std::ffi::c_void,
            },
            element_stride: ty.size(),
            total_size_bytes: size_bytes,
        })
    }

    fn destroy_buffer(&self, buffer: luisa_compute_api_types::Buffer) {
        unsafe {
            let ptr = buffer.0 as *mut BufferImpl;
            drop(Box::from_raw(ptr));
        }
    }

    fn create_texture(
        &self,
        format: luisa_compute_api_types::PixelFormat,
        dimension: u32,
        width: u32,
        height: u32,
        depth: u32,
        mipmap_levels: u32,
    ) -> super::Result<luisa_compute_api_types::CreatedResourceInfo> {
        let storage = format.storage();

        let texture = TextureImpl::new(
            dimension as u8,
            [width, height, depth],
            storage,
            mipmap_levels as u8,
        );
        let data = texture.data;
        let ptr = Box::into_raw(Box::new(texture));
        Ok(CreatedResourceInfo {
            handle: ptr as u64,
            native_handle: data as *mut std::ffi::c_void,
        })
    }

    fn destroy_texture(&self, texture: luisa_compute_api_types::Texture) {
        unsafe {
            let texture = texture.0 as *mut TextureImpl;
            drop(Box::from_raw(texture));
        }
    }
    fn create_bindless_array(
        &self,
        size: usize,
    ) -> super::Result<luisa_compute_api_types::CreatedResourceInfo> {
        let bindless_array = BindlessArrayImpl {
            buffers: vec![defs::BufferView::default(); size],
            tex2ds: vec![defs::Texture::default(); size],
            tex3ds: vec![defs::Texture::default(); size],
        };
        let ptr = Box::into_raw(Box::new(bindless_array));
        Ok(CreatedResourceInfo {
            handle: ptr as u64,
            native_handle: ptr as *mut std::ffi::c_void,
        })
    }

    fn destroy_bindless_array(&self, array: luisa_compute_api_types::BindlessArray) {
        unsafe {
            let ptr = array.0 as *mut BindlessArrayImpl;
            drop(Box::from_raw(ptr));
        }
    }

    fn create_stream(
        &self,
        _tag: api::StreamTag,
    ) -> super::Result<luisa_compute_api_types::CreatedResourceInfo> {
        let stream = Box::into_raw(Box::new(StreamImpl::new(self.shared_pool.clone())));
        Ok(CreatedResourceInfo {
            handle: stream as u64,
            native_handle: stream as *mut std::ffi::c_void,
        })
    }

    fn destroy_stream(&self, stream: luisa_compute_api_types::Stream) {
        unsafe {
            let stream = stream.0 as *mut StreamImpl;
            drop(Box::from_raw(stream));
        }
    }

    fn synchronize_stream(&self, stream: luisa_compute_api_types::Stream) -> super::Result<()> {
        unsafe {
            let stream = stream.0 as *mut StreamImpl;
            (*stream).synchronize();
            Ok(())
        }
    }

    fn dispatch(
        &self,
        stream_: luisa_compute_api_types::Stream,
        command_list: &[luisa_compute_api_types::Command],
        callback: (extern "C" fn(*mut u8), *mut u8),
    ) -> super::Result<()> {
        unsafe {
            let stream = &*(stream_.0 as *mut StreamImpl);
            let command_list = command_list.to_vec();
            stream.enqueue(
                move || {
                    let stream = &*(stream_.0 as *mut StreamImpl);
                    stream.dispatch(&command_list)
                },
                callback,
            );
            Ok(())
        }
    }
    fn create_shader(
        &self,
        kernel: CArc<luisa_compute_ir::ir::KernelModule>,
        _options: &api::ShaderOption,
    ) -> super::Result<luisa_compute_api_types::CreatedShaderInfo> {
        // let debug =
        //     luisa_compute_ir::ir::debug::luisa_compute_ir_dump_human_readable(&kernel.module);
        // let debug = std::ffi::CString::new(debug.as_ref()).unwrap();
        // println!("{}", debug.to_str().unwrap());
        let tic = std::time::Instant::now();
        let gened_src = luisa_compute_ir::codegen::generic_cpp::CpuCodeGen::run(&kernel);
        info!(
            "kernel source generated in {:.3}ms",
            (std::time::Instant::now() - tic).as_secs_f64() * 1e3
        );
        // println!("{}", gened_src);
        let lib_path = shader::compile(gened_src).unwrap();
        let mut captures = vec![];
        let mut custom_ops = vec![];
        unsafe {
            for c in kernel.captures.as_ref() {
                captures.push(convert_capture(*c));
            }
            for op in kernel.cpu_custom_ops.as_ref() {
                custom_ops.push(defs::CpuCustomOp {
                    func: op.func,
                    data: op.data,
                });
            }
        }
        let shader = Box::new(shader::ShaderImpl::new(
            lib_path,
            captures,
            custom_ops,
            kernel.block_size,
        ));
        let shader = Box::into_raw(shader);
        Ok(luisa_compute_api_types::CreatedShaderInfo {
            resource: CreatedResourceInfo {
                handle: shader as u64,
                native_handle: shader as *mut std::ffi::c_void,
            },
            block_size: kernel.block_size,
        })
    }
    fn shader_cache_dir(
        &self,
        shader: luisa_compute_api_types::Shader,
    ) -> Option<std::path::PathBuf> {
        unsafe {
            let shader = shader.0 as *mut shader::ShaderImpl;
            Some((*shader).dir.clone())
        }
    }
    fn destroy_shader(&self, shader: luisa_compute_api_types::Shader) {
        unsafe {
            let shader = shader.0 as *mut shader::ShaderImpl;
            drop(Box::from_raw(shader));
        }
    }

    fn create_event(&self) -> super::Result<luisa_compute_api_types::CreatedResourceInfo> {
        todo!()
    }

    fn destroy_event(&self, _event: luisa_compute_api_types::Event) {
        todo!()
    }

    fn signal_event(&self, _event: api::Event, _stream: api::Stream) {
        todo!()
    }

    fn wait_event(
        &self,
        _event: luisa_compute_api_types::Event,
        _stream: api::Stream,
    ) -> super::Result<()> {
        todo!()
    }

    fn synchronize_event(&self, _event: luisa_compute_api_types::Event) -> super::Result<()> {
        todo!()
    }
    fn create_mesh(&self, option: AccelOption) -> super::Result<api::CreatedResourceInfo> {
        unsafe {
            let mesh = Box::new(MeshImpl::new(
                option.hint,
                option.allow_compaction,
                option.allow_update,
            ));
            let mesh = Box::into_raw(mesh);
            Ok(api::CreatedResourceInfo {
                handle: mesh as u64,
                native_handle: mesh as *mut std::ffi::c_void,
            })
        }
    }
    fn destroy_mesh(&self, mesh: api::Mesh) {
        unsafe {
            let mesh = mesh.0 as *mut MeshImpl;
            drop(Box::from_raw(mesh));
        }
    }
    fn create_procedural_primitive(
        &self,
        _option: api::AccelOption,
    ) -> crate::Result<api::CreatedResourceInfo> {
        todo!()
    }
    fn destroy_procedural_primitive(&self, _primitive: api::ProceduralPrimitive) {
        todo!()
    }
    fn create_accel(&self, _option: AccelOption) -> super::Result<api::CreatedResourceInfo> {
        unsafe {
            let accel = Box::new(AccelImpl::new());
            let accel = Box::into_raw(accel);
            Ok(api::CreatedResourceInfo {
                handle: accel as u64,
                native_handle: accel as *mut std::ffi::c_void,
            })
        }
    }
    fn destroy_accel(&self, accel: api::Accel) {
        unsafe {
            let accel = accel.0 as *mut AccelImpl;
            drop(Box::from_raw(accel));
        }
    }
    fn create_swapchain(
        &self,
        window_handle: u64,
        stream_handle: api::Stream,
        width: u32,
        height: u32,
        allow_hdr: bool,
        vsync: bool,
        back_buffer_size: u32,
    ) -> crate::Result<api::CreatedSwapchainInfo> {
        let ctx = self.swapchain_context.read();
        let ctx = ctx
            .as_ref()
            .unwrap_or_else(|| panic!("swapchain context is not initialized"));
        unsafe {
            let sc_ctx = (ctx.create_cpu_swapchain)(
                window_handle,
                stream_handle.0,
                width,
                height,
                allow_hdr,
                vsync,
                back_buffer_size,
            );
            let storage = (ctx.cpu_swapchain_storage)(sc_ctx);
            Ok(api::CreatedSwapchainInfo {
                resource: api::CreatedResourceInfo {
                    handle: sc_ctx as u64,
                    native_handle: sc_ctx as *mut std::ffi::c_void,
                },
                storage: std::mem::transmute(storage as u32),
            })
        }
    }
    fn destroy_swapchain(&self, swap_chain: api::Swapchain) {
        let ctx = self.swapchain_context.read();
        let ctx = ctx
            .as_ref()
            .unwrap_or_else(|| panic!("swapchain context is not initialized"));
        unsafe {
            (ctx.destroy_cpu_swapchain)(swap_chain.0 as *mut c_void);
        }
    }
    fn present_display_in_stream(
        &self,
        stream_handle: api::Stream,
        swapchain_handle: api::Swapchain,
        image_handle: api::Texture,
    ) {
        let ctx = self.swapchain_context.read();
        let ctx = ctx
            .as_ref()
            .unwrap_or_else(|| panic!("swapchain context is not initialized"));
        unsafe {
            let stream = &*(stream_handle.0 as *mut StreamImpl);
            let img = &*(image_handle.0 as *mut TextureImpl);
            let storage = (ctx.cpu_swapchain_storage)(swapchain_handle.0 as *mut c_void);
            let storage: api::PixelStorage = std::mem::transmute(storage as u32);
            assert_eq!(storage, img.storage);
            extern "C" fn empty_callback(_: *mut u8) {}
            let present = ctx.cpu_swapchain_present;
            stream.enqueue(
                move || {
                    let pixels = img.view(0).copy_to_vec_par_2d();

                    present(
                        swapchain_handle.0 as *mut c_void,
                        pixels.as_ptr() as *const c_void,
                        pixels.len() as u64,
                    );
                },
                (empty_callback, std::ptr::null_mut()),
            )
        }
    }
}
impl RustBackend {
    pub fn new() -> Arc<Self> {
        let num_threads = match std::env::var("LUISA_NUM_THREADS") {
            Ok(s) => s.parse::<usize>().unwrap(),
            Err(_) => std::thread::available_parallelism().unwrap().get(),
        };

        Arc::new(RustBackend {
            shared_pool: Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .unwrap(),
            ),
            swapchain_context: RwLock::new(None),
        })
    }
}
