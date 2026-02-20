mod collective {
    pub mod helpers;

    mod allreduce;
    mod barrier;
    mod broadcast;
    mod compressed;
    mod exchange;
    mod iov;
    mod nonblocking;
    mod reduce;
    mod rpc;
    mod scan;
    mod split;
    mod tagged;
}
