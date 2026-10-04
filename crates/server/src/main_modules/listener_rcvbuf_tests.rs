// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Tests of the listener receive-buffer sizing: the plan for each
//! kind of host, the `tcp_rmem` parse, inheritance by accepted sockets, and
//! (ignored; it needs a host whose `tcp_rmem` default is below the kernel's)
//! the request-body delay itself, through the real bind.
//!
//! Owner: server (HTTP layer).
//! Invariants: none beyond the types.

use super::{
    KERNEL_DEFAULT_TCP_RMEM_BYTES, LISTENER_RCVBUF_BYTES, RcvbufPlan, parse_tcp_rmem_default, plan,
};

#[test]
fn a_host_at_or_above_the_kernel_default_keeps_autotuning() {
    for d in [KERNEL_DEFAULT_TCP_RMEM_BYTES, 6 * 1024 * 1024] {
        assert_eq!(plan(Some(d)), RcvbufPlan::Keep { default_bytes: d });
    }
}

#[test]
fn a_host_below_the_kernel_default_gets_the_listener_buffer() {
    // 2026-10-03: 87380 is the value measured to delay 32k-token bodies by 200 ms.
    for d in [4096, 87_380, KERNEL_DEFAULT_TCP_RMEM_BYTES - 1] {
        assert_eq!(
            plan(Some(d)),
            RcvbufPlan::Set {
                default_bytes: d,
                bytes: LISTENER_RCVBUF_BYTES
            }
        );
    }
}

#[test]
fn an_unread_default_leaves_the_listener_alone() {
    assert_eq!(plan(None), RcvbufPlan::Unknown);
}

#[test]
fn the_default_is_the_middle_field_of_tcp_rmem() {
    assert_eq!(
        parse_tcp_rmem_default("4096\t87380\t134217728\n").unwrap(),
        87_380
    );
    assert_eq!(
        parse_tcp_rmem_default("4096 131072 33554432").unwrap(),
        131_072
    );
}

#[test]
fn a_malformed_tcp_rmem_is_an_error_not_a_guess() {
    for text in [
        "",
        "4096 87380",
        "4096 87380 1 2",
        "4096 lots 134217728",
        "-1 -1 -1",
    ] {
        assert!(
            parse_tcp_rmem_default(text).is_err(),
            "{text:?} must not parse"
        );
    }
}

/// 2026-10-03: The accepted socket's `SO_RCVBUF`, read through `socket2`.
#[cfg(target_os = "linux")]
fn accepted_rcvbuf(listener: &tokio::net::TcpListener, rt: &tokio::runtime::Runtime) -> usize {
    let addr = listener.local_addr().expect("addr");
    rt.block_on(async {
        let (_client, accepted) = tokio::join!(tokio::net::TcpStream::connect(addr), async {
            listener.accept().await.expect("accept").0
        });
        socket2::SockRef::from(&accepted)
            .recv_buffer_size()
            .expect("SO_RCVBUF")
    })
}

#[cfg(target_os = "linux")]
#[test]
fn accepted_sockets_inherit_the_listener_buffer() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let listener = rt
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let before = accepted_rcvbuf(&listener, &rt);
    let effective = super::set_listener_rcvbuf(&listener, LISTENER_RCVBUF_BYTES).expect("set");
    let after = accepted_rcvbuf(&listener, &rt);
    assert_eq!(
        after, effective,
        "a socket accepted after the set carries the listener's buffer"
    );
    assert_ne!(
        after, before,
        "the set changed what accepted sockets get (before {before} B)"
    );
}

/// 2026-10-03: Time one HTTP/1.1 POST of `body_bytes` to `addr`, the way the
/// bench client sends it (one write, `TCP_NODELAY`, `Connection: close`), until
/// the response has been read to EOF.
#[cfg(target_os = "linux")]
async fn post_ms(addr: std::net::SocketAddr, body_bytes: usize) -> f64 {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    let body = "a".repeat(body_bytes);
    let request = format!(
        "POST /echo HTTP/1.1\r\nHost: {addr}\r\nContent-Type: text/plain\r\n\
         Connection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let mut sock = tokio::net::TcpStream::connect(addr).await.expect("connect");
    sock.set_nodelay(true).expect("nodelay");
    let started = std::time::Instant::now();
    sock.write_all(request.as_bytes()).await.expect("write");
    let mut response = Vec::new();
    sock.read_to_end(&mut response).await.expect("read");
    assert!(
        response.starts_with(b"HTTP/1.1 200"),
        "the server answered: {:?}",
        String::from_utf8_lossy(&response[..response.len().min(80)])
    );
    started.elapsed().as_secs_f64() * 1000.0
}

/// 2026-10-03: The slowest of five posts of a 32k-token-sized body (127 KB)
/// to an axum app served on `listener`.
#[cfg(target_os = "linux")]
async fn slowest_post_ms(listener: tokio::net::TcpListener) -> f64 {
    let addr = listener.local_addr().expect("addr");
    let app = axum::Router::new().route(
        "/echo",
        axum::routing::post(|body: String| async move { body.len().to_string() }),
    );
    let server = tokio::spawn(async move { axum::serve(listener, app).await });
    let mut slowest = 0.0_f64;
    for _ in 0..5 {
        slowest = slowest.max(post_ms(addr, 127_000).await);
    }
    server.abort();
    slowest
}

/// 2026-10-03: The delay this module removes, end to end. It needs a host whose
/// `tcp_rmem` default is below the kernel's, which a CI runner is not, and it
/// refuses to run anywhere else rather than pass without testing anything. Run
/// it on such a host with
/// `cargo test -p metrale-server listener_rcvbuf -- --ignored`.
#[cfg(target_os = "linux")]
#[ignore = "needs net.ipv4.tcp_rmem default < 131072 on the host; see the doc comment"]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_large_body_is_not_delayed_on_a_host_with_a_small_tcp_rmem_default() {
    let default = super::read_tcp_rmem_default().expect("read tcp_rmem");
    assert!(
        default < KERNEL_DEFAULT_TCP_RMEM_BYTES,
        "this host's tcp_rmem default is {default} B; the test needs one below \
         {KERNEL_DEFAULT_TCP_RMEM_BYTES} B to reproduce the delay"
    );

    // 2026-10-03: Control: a listener bound without the sizing shows the delay,
    // so the fixed leg below is not passing on a host that never had it.
    let plain = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let unfixed = slowest_post_ms(plain).await;
    assert!(
        unfixed >= 150.0,
        "control: without the sizing a 127 KB body should wait on the 200 ms window \
         probe; the slowest post took {unfixed:.1} ms"
    );

    // 2026-10-03: The production bind, which sizes the listener.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .expect("probe a free port")
        .local_addr()
        .expect("addr")
        .port();
    let host = crate::main_modules::model_host::ModelHost::empty();
    let fixed_listener =
        crate::main_modules::serve_router::bind_and_announce(&host, "127.0.0.1", port)
            .await
            .expect("bind");
    let fixed = slowest_post_ms(fixed_listener).await;
    assert!(
        fixed < 20.0,
        "with the sizing the slowest 127 KB post took {fixed:.1} ms (unfixed {unfixed:.1} ms)"
    );
}
