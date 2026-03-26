from turboagents.proxy.dashboard import DashboardState, describe_dashboard


def test_dashboard_state_updates() -> None:
    state = DashboardState()
    state.update(status="ok", tok_s=42)
    snap = state.snapshot()
    assert snap["status"] == "ok"
    assert snap["tok_s"] == 42
    assert "status: ok" in describe_dashboard(state)

