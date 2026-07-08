#[test]
fn per_rank_kv_blocks_uses_connected_engine_capacity() {
    let mut rank = vllm_engine_core_client::mock_engine::default_ready_response();
    rank.num_gpu_blocks = 1_000;
    rank.data_parallel_size = 4;
    assert_eq!(super::super::per_rank_kv_blocks(&[&rank]), 1_000);

    let mut smaller = rank.clone();
    smaller.num_gpu_blocks = 900;
    assert_eq!(super::super::per_rank_kv_blocks(&[&rank, &smaller]), 900);
    assert_eq!(super::super::per_rank_kv_blocks(&[]), 0);
}

#[test]
fn offset_endpoint_port_matches_vllm_convention() {
    assert_eq!(
        super::super::offset_endpoint_port("tcp://*:5557", 0),
        "tcp://*:5557"
    );
    assert_eq!(
        super::super::offset_endpoint_port("tcp://*:5557", 1),
        "tcp://*:5558"
    );
    assert_eq!(
        super::super::offset_endpoint_port("tcp://127.0.0.1:5557", 7),
        "tcp://127.0.0.1:5564"
    );
    assert_eq!(
        super::super::offset_endpoint_port("inproc://cache", 3),
        "inproc://cache_dp3"
    );
    assert_eq!(super::super::offset_endpoint_port("", 5), "");
}
