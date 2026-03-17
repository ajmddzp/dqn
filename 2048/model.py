# -*- coding: UTF-8 -*-
import random
import numpy as np
import torch

try:
    import pygame
except ImportError:
    pygame = None


BOARD_SIZE = 4
ENABLE_RENDER = False
RENDER_FPS = 30
_warned_no_pygame = False

gameMap = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
score = 0
step = 0
isGameOver = False
_prev_max_tile = 0
_render_clock = None

screen = None

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


def _ensure_render_initialized():
    global screen, _render_clock
    if pygame is None:
        return False
    if not pygame.get_init():
        pygame.init()
    if screen is None:
        screen = pygame.display.set_mode((410, 500))
        pygame.display.set_caption("2048 RL")
    if _render_clock is None:
        _render_clock = pygame.time.Clock()
    return True


def set_render(enabled=True, fps=30):
    global ENABLE_RENDER, RENDER_FPS, _warned_no_pygame
    requested = bool(enabled)
    RENDER_FPS = int(fps)
    if requested and pygame is None:
        ENABLE_RENDER = False
        if not _warned_no_pygame:
            print("Render disabled: pygame is not installed. Install with: pip install pygame")
            _warned_no_pygame = True
        return
    ENABLE_RENDER = requested
    if ENABLE_RENDER:
        _ensure_render_initialized()


def _encode_state():
    board = np.array(gameMap, dtype=np.float32)
    encoded = np.zeros_like(board, dtype=np.float32)
    mask = board > 0
    encoded[mask] = np.log2(board[mask])
    return torch.tensor(encoded, dtype=torch.float32).flatten()


def _empty_positions():
    positions = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if gameMap[i][j] == 0:
                positions.append((i, j))
    return positions


def hasEmptyPosition():
    return len(_empty_positions()) > 0


def getRandomPos():
    empties = _empty_positions()
    if not empties:
        raise RuntimeError("gameMap is full, cannot generate new tile")
    return random.choice(empties)


def _spawn_tile():
    if not hasEmptyPosition():
        return
    i, j = getRandomPos()
    gameMap[i][j] = 4 if random.random() < 0.1 else 2


def makeMap():
    _spawn_tile()
    _spawn_tile()


def _can_move():
    if hasEmptyPosition():
        return True
    board = np.array(gameMap, dtype=np.int32)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if i + 1 < BOARD_SIZE and board[i, j] == board[i + 1, j]:
                return True
            if j + 1 < BOARD_SIZE and board[i, j] == board[i, j + 1]:
                return True
    return False


def referee():
    if any(2048 in row for row in gameMap):
        return 1
    return 0 if _can_move() else 2


def _merge_left_line(line):
    values = [v for v in line if v != 0]
    merged = []
    gained = 0
    i = 0
    while i < len(values):
        if i + 1 < len(values) and values[i] == values[i + 1]:
            new_value = values[i] * 2
            merged.append(new_value)
            gained += new_value
            i += 2
        else:
            merged.append(values[i])
            i += 1
    merged += [0] * (BOARD_SIZE - len(merged))
    return merged, gained


def _move_left(board):
    moved = False
    total_gain = 0
    new_rows = []
    for row in board.tolist():
        merged_row, gained = _merge_left_line(row)
        if merged_row != row:
            moved = True
        total_gain += gained
        new_rows.append(merged_row)
    return np.array(new_rows, dtype=np.int32), total_gain, moved


def _apply_action(act):
    board = np.array(gameMap, dtype=np.int32)
    moved = False
    gained = 0

    if act == 1:  # left
        board, gained, moved = _move_left(board)
    elif act == 3:  # right
        reversed_board = np.fliplr(board)
        reversed_board, gained, moved = _move_left(reversed_board)
        board = np.fliplr(reversed_board)
    elif act == 0:  # up
        transposed = board.T
        transposed, gained, moved = _move_left(transposed)
        board = transposed.T
    elif act == 2:  # down
        transposed = board.T
        reversed_transposed = np.fliplr(transposed)
        reversed_transposed, gained, moved = _move_left(reversed_transposed)
        board = np.fliplr(reversed_transposed).T

    return board.tolist(), gained, moved


def show():
    if not ENABLE_RENDER:
        return
    if not _ensure_render_initialized():
        return

    screen.fill(BG_COLOR)
    side = 90
    spacing = 10
    top_margin = 10

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            left = j * (side + spacing) + 10
            top = i * (side + spacing) + top_margin
            value = gameMap[i][j]
            color = EMPTY_COLOR if value == 0 else TILE_COLORS.get(value, (60, 58, 50))
            pygame.draw.rect(screen, color, (left, top, side, side), border_radius=8)
            if value != 0:
                digits = len(str(value))
                font_size = max(24, 56 - digits * 8)
                font = pygame.font.SysFont("Arial", font_size, bold=True)
                text_color = DARK_TEXT_COLOR if value <= 4 else LIGHT_TEXT_COLOR
                text = font.render(str(value), True, text_color)
                text_x = left + (side - text.get_width()) / 2
                text_y = top + (side - text.get_height()) / 2
                screen.blit(text, (text_x, text_y))

    info_font = pygame.font.SysFont("Arial", 22, bold=True)
    info_text = info_font.render(f"Score: {score}   Step: {step}", True, (250, 248, 239))
    screen.blit(info_text, (10, 420))

    pygame.display.flip()
    _render_clock.tick(RENDER_FPS)


def action(val):
    return val


def reset():
    global score, step, gameMap, isGameOver, _prev_max_tile
    score = 0
    step = 0
    isGameOver = False
    gameMap = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    makeMap()
    _prev_max_tile = int(np.max(gameMap))
    show()
    return _encode_state()


def RL_step(act):
    global score, step, gameMap, isGameOver, _prev_max_tile

    if ENABLE_RENDER:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                isGameOver = True
                return _encode_state(), -10.0, True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                isGameOver = True
                return _encode_state(), -10.0, True

    act = int(act) % 4
    prev_score = score
    new_map, merge_gain, moved = _apply_action(act)

    if moved:
        gameMap = new_map
        score += merge_gain
        step += 1
        _spawn_tile()

    done = not _can_move()
    isGameOver = done
    current_max = int(np.max(gameMap))

    # Reward shaping:
    # 1) merge gain drives score growth
    # 2) empty cells encourage keeping space
    # 3) reaching a new max tile gives sparse bonus
    if moved:
        empty_count = sum(1 for row in gameMap for v in row if v == 0)
        merge_reward = (score - prev_score) / 16.0
        empty_reward = 0.05 * empty_count
        max_tile_bonus = 0.0
        if current_max > _prev_max_tile:
            max_tile_bonus = float(np.log2(current_max) - np.log2(max(_prev_max_tile, 2)))
        reward = merge_reward + empty_reward + max_tile_bonus
    else:
        reward = -0.5

    _prev_max_tile = max(_prev_max_tile, current_max)
    if done:
        reward -= 10.0

    show()
    return _encode_state(), float(reward), bool(done)
