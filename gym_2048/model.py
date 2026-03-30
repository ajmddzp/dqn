from __future__ import annotations

try:
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    import gym
import gym_2048  # noqa: F401  # trigger env registration
import numpy as np
import pygame


BG_COLOR = (187, 173, 160)
EMPTY_COLOR = (205, 193, 180)
TILE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (247, 96, 63),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}
LIGHT_TEXT_COLOR = (249, 246, 242)
DARK_TEXT_COLOR = (119, 110, 101)

WINDOW_WIDTH = 420
WINDOW_HEIGHT = 520
CELL_SIZE = 90
CELL_GAP = 10
BOARD_LEFT = 10
BOARD_TOP = 10

# user_input -> env action
# input: 0=up, 1=down, 2=left, 3=right
# env action: 0=left, 1=up, 2=right, 3=down
USER_TO_ENV_ACTION = {0: 1, 1: 3, 2: 0, 3: 2}
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


def to_env_action(user_action: int) -> int:
    """Map user numeric input (0..3) to gym-2048 action id."""
    if user_action not in USER_TO_ENV_ACTION:
        raise ValueError("user_action must be one of 0, 1, 2, 3")
    return USER_TO_ENV_ACTION[user_action]


def draw_board(screen, board, score_hint, step_id):
    screen.fill(BG_COLOR)

    for i in range(4):
        for j in range(4):
            value = int(board[i, j])
            left = BOARD_LEFT + j * (CELL_SIZE + CELL_GAP)
            top = BOARD_TOP + i * (CELL_SIZE + CELL_GAP)
            color = EMPTY_COLOR if value == 0 else TILE_COLORS.get(value, (60, 58, 50))
            pygame.draw.rect(
                screen, color, (left, top, CELL_SIZE, CELL_SIZE), border_radius=8
            )

            if value != 0:
                digits = len(str(value))
                font_size = max(24, 56 - digits * 8)
                font = pygame.font.SysFont("Arial", font_size, bold=True)
                text_color = DARK_TEXT_COLOR if value <= 4 else LIGHT_TEXT_COLOR
                text = font.render(str(value), True, text_color)
                text_x = left + (CELL_SIZE - text.get_width()) / 2
                text_y = top + (CELL_SIZE - text.get_height()) / 2
                screen.blit(text, (text_x, text_y))

    info_font = pygame.font.SysFont("Arial", 22, bold=True)
    info_text = info_font.render(
        f"BoardSum: {score_hint}   Step: {step_id}", True, (250, 248, 239)
    )
    screen.blit(info_text, (10, 430))
    pygame.display.flip()


class Game2048:
    """Simple interface: reset() + step(0..3) + close()."""

    def __init__(
        self, seed=42, fps=30, window_title="2048 Gymnasium", render_enabled=True
    ):
        pygame.init()
        self.window_title = str(window_title)
        self.render_enabled = bool(render_enabled)
        self.screen = None
        if self.render_enabled:
            self._ensure_window()
        self.clock = pygame.time.Clock()

        self.env = gym.make("2048-extended-v2")
        self.fps = int(fps)
        self.closed = False

        self.board = None
        self.info = {}
        self.done = False
        self.step_id = 0
        self.reset(seed=seed)

    def _ensure_window(self):
        if not pygame.display.get_init():
            pygame.display.init()
        if self.screen is None:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption(self.window_title)

    def _process_events(self):
        if not self.render_enabled or not pygame.display.get_init():
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.done = True

    def _compose_info(self, base_info):
        info = dict(base_info)
        info["empty_cells"] = int(np.count_nonzero(self.board == 0))
        info["max_tile"] = int(np.max(self.board))
        info["step_id"] = int(self.step_id)
        return info

    def reset(self, seed=None):
        if self.closed:
            raise RuntimeError("Game2048 has been closed.")
        try:
            reset_out = self.env.reset(seed=seed)
        except TypeError:
            reset_out = self.env.reset()

        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            self.board, base_info = reset_out
        else:
            self.board, base_info = reset_out, {}

        self.done = False
        self.step_id = 0
        self.info = self._compose_info(base_info)
        self.render()
        return self.board.copy(), dict(self.info)

    def step(self, user_action):
        """
        Execute one step by user action code.

        Args:
            user_action (int): user action id in {0, 1, 2, 3}
                0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.

        Returns:
            tuple[np.ndarray, float, bool, dict]:
                board (np.ndarray):
                    current board after this step, shape (4, 4), returned as a copy.
                reward (float):
                    immediate reward returned by env for this step.
                    If user closes the window/presses ESC, this returns 0.0.
                done (bool):
                    whether current episode is finished.
                    True when terminated/truncated or closed by user.
                info (dict):
                    merged runtime info for logging/training, including:
                    - end_value / max_block / is_success (from env)
                    - empty_cells / max_tile / step_id (computed here)
                    - user_action / action_name (added here)
                    - closed_by_user=True (only in early-close branch)
        """
        if self.closed:
            raise RuntimeError("Game2048 has been closed.")

        self._process_events()
        if self.done:
            info = dict(self.info)
            info["closed_by_user"] = True
            return self.board.copy(), 0.0, True, info

        env_action = to_env_action(int(user_action))
        step_out = self.env.step(env_action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            self.board, reward, terminated, truncated, base_info = step_out
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            self.board, reward, done, base_info = step_out
            terminated, truncated = bool(done), False
        else:
            raise RuntimeError(
                "Unexpected env.step() return format; expected 4 or 5 values."
            )

        self.step_id += 1
        self.done = bool(terminated or truncated)
        self.info = self._compose_info(base_info)
        self.info["user_action"] = int(user_action)
        self.info["action_name"] = ACTION_NAMES[int(user_action)]

        self.render()
        if self.render_enabled:
            self.clock.tick(self.fps)
        # Return a stable 4-tuple interface for training loops.
        return self.board.copy(), float(reward), self.done, dict(self.info)

    def render(self):
        if not self.render_enabled:
            return
        self._ensure_window()
        score_hint = int(self.info.get("end_value", np.sum(self.board)))
        draw_board(self.screen, self.board, score_hint, self.step_id)

    def set_render_enabled(self, enabled: bool):
        enabled = bool(enabled)
        if enabled == self.render_enabled:
            return

        self.render_enabled = enabled
        if self.render_enabled:
            self._ensure_window()
            self.render()
        else:
            if pygame.display.get_init():
                pygame.display.quit()
            self.screen = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.env.close()
        if pygame.display.get_init():
            pygame.display.quit()
        pygame.quit()


def run(game: Game2048, action: int):
    """Single-step helper for external loops."""
    if action not in {0, 1, 2, 3}:
        raise ValueError("action must be one of 0, 1, 2, 3")

    return game.step(action)


def play_manual():
    game = Game2048(seed=42, fps=30, window_title="2048 Manual Control")
    try:
        done = False
        while not done:
            raw = input(
                "Input action [0=up, 1=down, 2=left, 3=right, q=quit]: "
            ).strip()
            if raw.lower() in {"q", "quit", "exit"}:
                break
            if raw not in {"0", "1", "2", "3"}:
                print("Invalid input, please enter 0/1/2/3 (or q to quit).")
                continue

            done, _ = run(game, int(raw))

        print(f"Total Moves: {game.step_id}")
    finally:
        game.close()


if __name__ == "__main__":
    play_manual()
