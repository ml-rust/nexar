use std::collections::HashMap;
use std::sync::Arc;

/// Type alias for RPC handler functions.
///
/// Receives the serialized arguments and returns serialized response.
pub type RpcHandler = Arc<dyn Fn(&[u8]) -> Vec<u8> + Send + Sync>;

/// Registry mapping function IDs to handler closures.
pub struct RpcRegistry {
    handlers: HashMap<u16, RpcHandler>,
}

impl RpcRegistry {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register a handler for a given function ID.
    pub fn register(&mut self, fn_id: u16, handler: RpcHandler) {
        self.handlers.insert(fn_id, handler);
    }

    /// Look up a handler by function ID.
    pub fn get(&self, fn_id: u16) -> Option<&RpcHandler> {
        self.handlers.get(&fn_id)
    }

    /// Check if a handler is registered.
    pub fn contains(&self, fn_id: u16) -> bool {
        self.handlers.contains_key(&fn_id)
    }
}

impl Default for RpcRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_lookup() {
        let mut reg = RpcRegistry::new();

        let handler: RpcHandler = Arc::new(|args: &[u8]| {
            let mut result = args.to_vec();
            result.push(0xFF);
            result
        });

        reg.register(42, handler);
        assert!(reg.contains(42));
        assert!(!reg.contains(99));

        let h = reg.get(42).unwrap();
        let result = h(&[1, 2, 3]);
        assert_eq!(result, vec![1, 2, 3, 0xFF]);
    }

    #[test]
    fn test_overwrite_handler() {
        let mut reg = RpcRegistry::new();
        reg.register(1, Arc::new(|_| vec![0]));
        reg.register(1, Arc::new(|_| vec![1]));

        let h = reg.get(1).unwrap();
        assert_eq!(h(&[]), vec![1]);
    }
}
