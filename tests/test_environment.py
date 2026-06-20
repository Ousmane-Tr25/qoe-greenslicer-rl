from qoe_greenslicer_rl.environment import NetworkSlicingEnv
from qoe_greenslicer_rl.experiment import train_agent


def test_environment_step_returns_valid_values():
    env = NetworkSlicingEnv(seed=1)
    state = env.reset()
    next_state, reward, done, info = env.step(0)
    assert len(state) == 3
    assert len(next_state) == 3
    assert isinstance(reward, float)
    assert "qoe" in info
    assert done is False


def test_agent_can_be_trained_for_small_number_of_episodes():
    agent = train_agent(episodes=3, seed=1)
    assert len(agent.q_table) > 0
