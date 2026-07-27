// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::body::{Body, Bytes, HttpBody};
use axum::extract::{MatchedPath, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http_body::{Frame, SizeHint};

use crate::state::{AppState, EngineWorkGuard};

/// Pure frontend metadata endpoints remain available while draining. Every
/// other matched route is admitted as engine work by default, so new engine
/// routes cannot accidentally bypass process-wide draining.
const FRONTEND_ONLY_HANDLERS: &[&str] = &[
    "/health",
    "/metrics",
    "/load",
    "/version",
    "/v1/models",
    "/server_info",
    "/tokenize",
    "/detokenize",
];

/// Track frontend-local in-flight inference requests for the `/load` endpoint.
pub async fn track_server_load(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    let Some(handler) = req.extensions().get::<MatchedPath>().map(|path| path.as_str()) else {
        return next.run(req).await;
    };

    if FRONTEND_ONLY_HANDLERS.contains(&handler) {
        return next.run(req).await;
    }

    let Some(guard) = state.try_admit_engine_work() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "engine is draining").into_response();
    };
    let response = next.run(req).await;

    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(LoadTrackedBody {
            inner: body,
            _guard: guard,
        }),
    )
}

/// A wrapper around response bodies that tracks server load by holding a
/// guard, which will decrement the load when the body is fully
/// consumed and dropped.
struct LoadTrackedBody {
    inner: Body,
    _guard: EngineWorkGuard,
}

// Simply delegate all `HttpBody` methods to the inner body.
impl HttpBody for LoadTrackedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}
