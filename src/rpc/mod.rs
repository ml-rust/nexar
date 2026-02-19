pub(crate) mod dispatcher;
pub(crate) mod registry;

pub use dispatcher::RpcDispatcher;
pub use registry::{RpcHandler, RpcRegistry};
