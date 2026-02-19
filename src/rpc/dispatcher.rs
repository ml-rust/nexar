use crate::error::{NexarError, Result};
use crate::protocol::NexarMessage;
use crate::rpc::registry::RpcRegistry;
use crate::transport::PeerConnection;
use crate::types::Priority;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};

/// Handles incoming RPC requests by dispatching to registered handlers.
pub struct RpcDispatcher {
    registry: Arc<RwLock<RpcRegistry>>,
}

impl RpcDispatcher {
    pub fn new(registry: Arc<RwLock<RpcRegistry>>) -> Self {
        Self { registry }
    }

    /// Handle a single incoming RPC message and return the response.
    pub async fn dispatch(&self, _req_id: u64, fn_id: u16, payload: &[u8]) -> Result<Vec<u8>> {
        let reg = self.registry.read().await;
        let handler = reg
            .get(fn_id)
            .ok_or(NexarError::RpcNotRegistered { fn_id })?;
        let response = handler(payload);
        Ok(response)
    }

    /// Process an RPC request and send the response back to the caller.
    pub async fn handle_request(
        &self,
        peer: &PeerConnection,
        req_id: u64,
        fn_id: u16,
        payload: &[u8],
    ) -> Result<()> {
        let response_payload = self.dispatch(req_id, fn_id, payload).await?;

        let response = NexarMessage::RpcResponse {
            req_id,
            payload: response_payload,
        };
        peer.send_message(&response, Priority::Realtime).await
    }

    /// Run a background loop that reads incoming RPC requests from the router's
    /// `rpc_requests` channel, dispatches them to registered handlers, and sends
    /// responses back via `peer`.
    ///
    /// This task runs until the channel is closed (i.e., the router has shut down).
    /// Spawn this with `tokio::spawn` for each peer that may send RPC requests.
    pub async fn serve(
        &self,
        peer: &PeerConnection,
        incoming: &mut mpsc::Receiver<NexarMessage>,
    ) -> Result<()> {
        while let Some(msg) = incoming.recv().await {
            // Router guarantees only Rpc messages arrive on rpc_requests channel.
            if let NexarMessage::Rpc {
                req_id,
                fn_id,
                payload,
            } = msg
            {
                self.handle_request(peer, req_id, fn_id, &payload).await?;
            }
        }
        Ok(())
    }
}
