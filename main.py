import pygame
import chess
import sys
import chess.pgn
import time
import chess.engine
import requests

# Initialize pygame
pygame.init()


print("hi")
# Constants
WIDTH, HEIGHT = 1000, 600
ROWS, COLS = 8, 8
SQUARE_SIZE = HEIGHT // ROWS

# Colors
WHITE = pygame.Color("antiquewhite")
BLACK = pygame.Color("chocolate4")
BLUE = pygame.Color("blue")
GREEN = pygame.Color("green")
RED = pygame.Color("red")


positions_analyzed = 0  # Global variable to count analyzed positions
depth_searched = 0  # Global variable to track depth

piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 333,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 10000
}

def load_images():
    pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
    images = {}
    for piece in pieces:
        images[piece] = pygame.image.load(f'ChessPieces/{piece}.png')
        images[piece] = pygame.transform.scale(images[piece], (SQUARE_SIZE, SQUARE_SIZE))
    return images

# Draw the board
def draw_board(screen, player_color):
    colors = [WHITE, BLACK]
    for row in range(ROWS):
        for col in range(COLS):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

# Draw the pieces
def draw_pieces(screen, board, images, player_color):
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        if player_color == chess.WHITE:
            row, col = 7 - row, col  # Only invert the row if player is white
        else:
            row, col = row, 7 - col  # Only invert the column if player is black
        piece_symbol = piece.symbol()
        piece_color = 'w' if piece.color == chess.WHITE else 'b'
        piece_key = f'{piece_color}{piece_symbol.upper()}'
        screen.blit(images[piece_key], (col * SQUARE_SIZE, row * SQUARE_SIZE))

# Highlight squares
def highlight_squares(screen, board, selected_square, player_color):
    if selected_square is not None:
        row, col = divmod(selected_square, 8)
        if player_color == chess.WHITE:
            row, col = 7 - row, col  # Only invert the row if player is white
        else:
            row, col = row, 7 - col  # Only invert the column if player is black
        pygame.draw.rect(screen, BLUE, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)
        for move in board.legal_moves:
            if move.from_square == selected_square:
                to_row, to_col = divmod(move.to_square, 8)
                if player_color == chess.WHITE:
                    to_row, to_col = 7 - to_row, to_col  # Only invert the row if player is white
                else:
                    to_row, to_col = to_row, 7 - to_col  # Only invert the column if player is black
                pygame.draw.rect(screen, GREEN, pygame.Rect(to_col * SQUARE_SIZE, to_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

# Draw check highlight
def draw_check(screen, board, player_color):
    if board.is_check():
        king_square = board.king(board.turn)
        row, col = divmod(king_square, 8)
        if player_color == chess.WHITE:
            row, col = 7 - row, col  # Only invert the row if player is white
        else:
            row, col = row, 7 - col  # Only invert the column if player is black
        pygame.draw.rect(screen, RED, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

# Get square under mouse
def get_square_under_mouse(player_color):
    x, y = pygame.mouse.get_pos()
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    if player_color == chess.WHITE:
        row, col = 7 - row, col  # Only invert the row if player is white
    else:
        row, col = row, 7 - col  # Only invert the column if player is black
    return chess.square(col, row)


# Draw the side panel
def draw_side_panel(screen, depth_searched, bot_evaluation, positions_analyzed, num_moves):
    panel_width = WIDTH - HEIGHT
    panel_height = HEIGHT
    panel_x = HEIGHT

    pygame.draw.rect(screen, WHITE, (panel_x, 0, panel_width, panel_height))
    pygame.draw.rect(screen, BLACK, (panel_x, 0, panel_width, panel_height), 2)

    font = pygame.font.SysFont("Arial", 24)
    texts = [
        f"Depth Searched: {depth_searched}",
        f"Bot's Evaluation: {bot_evaluation/100}",
        f"Positions Analyzed: {positions_analyzed}",
        f"Number of Moves: {num_moves}"
    ]

    for i, text in enumerate(texts):
        text_surface = font.render(text, True, BLACK)
        screen.blit(text_surface, (panel_x + 10, 10 + i * 30))





def draw_game_over_popup(screen, winner, num_moves):
    popup_width, popup_height = 300, 200
    popup_x, popup_y = (WIDTH - popup_width) // 2, (HEIGHT - popup_height) // 2
    font = pygame.font.SysFont("Arial", 24)

    pygame.draw.rect(screen, WHITE, (popup_x, popup_y, popup_width, popup_height))
    pygame.draw.rect(screen, BLACK, (popup_x, popup_y, popup_width, popup_height), 2)

    winner_text = f"{winner} wins!"
    moves_text = f"Total Moves: {num_moves}"
    play_again_text = "Play Again? (Y/N)"

    winner_surface = font.render(winner_text, True, BLACK)
    moves_surface = font.render(moves_text, True, BLACK)
    play_again_surface = font.render(play_again_text, True, BLACK)

    screen.blit(winner_surface, (popup_x + (popup_width - winner_surface.get_width()) // 2, popup_y + 20))
    screen.blit(moves_surface, (popup_x + (popup_width - moves_surface.get_width()) // 2, popup_y + 60))
    screen.blit(play_again_surface, (popup_x + (popup_width - play_again_surface.get_width()) // 2, popup_y + 100))

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    return True
                elif event.key == pygame.K_n:
                    return False


piece_square_tables = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
       -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    ]
}




def evaluate_board(board):
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -float('inf')  # Black wins
        else:
            return float('inf')  # White wins
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0  # Draw

    value = 0
    for square, piece in board.piece_map().items():
        piece_value = piece_values[piece.piece_type]
        piece_square_table = piece_square_tables[piece.piece_type]
        piece_square_value = piece_square_table[square] if piece.color == chess.WHITE else piece_square_table[chess.square_mirror(square)]
        if piece.color == chess.WHITE:
            value += piece_value + piece_square_value
        else:
            value -= piece_value + piece_square_value

    return value







def score_move(board, move):

    score = 0
    from_square = move.from_square
    to_square = move.to_square
    moving_piece = board.piece_at(from_square)

    if board.is_capture(move):
        captured_piece = board.piece_at(to_square)
        if captured_piece:
            score += piece_values[captured_piece.piece_type] - piece_values[moving_piece.piece_type]

    if board.is_attacked_by(not board.turn, to_square):
        score -= piece_values[moving_piece.piece_type] / 2

    if move.promotion:
        score += piece_values[move.promotion]

    return score


def minimax(board, depth, alpha, beta, maximizing_player, start_time, time_limit):
    global positions_analyzed, depth_searched
    depth_searched = max(depth_searched, depth)

    if board.is_checkmate():
        return float('-inf') if board.turn == chess.WHITE else float('inf')
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0

    if time.time() - start_time >= time_limit:
        return evaluate_board(board)

    if depth == 0:
        return quiescence_search(board, alpha, beta, start_time, time_limit)

    legal_moves = list(board.legal_moves)
    positions_analyzed += len(legal_moves)
    legal_moves.sort(key=lambda move: score_move(board, move), reverse=True)

    if maximizing_player:
        max_eval = float('-inf')
        for move in legal_moves:
            if time.time() - start_time >= time_limit:
                break
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return float('inf')  # Immediate checkmate found
            eval = minimax(board, depth - 1, alpha, beta, False, start_time, time_limit)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            if time.time() - start_time >= time_limit:
                break
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return float('-inf')  # Immediate checkmate found
            eval = minimax(board, depth - 1, alpha, beta, True, start_time, time_limit)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval




def quiescence_search(board, alpha, beta, start_time, time_limit):
    if board.is_checkmate():
        return float('-inf') if board.turn == chess.WHITE else float('inf')
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0

    if time.time() - start_time >= time_limit:
        return evaluate_board(board)

    stand_pat = evaluate_board(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in board.legal_moves:
        if time.time() - start_time >= time_limit:
            break
        if board.is_capture(move) or move.promotion:
            board.push(move)
            score = -quiescence_search(board, -beta, -alpha, start_time, time_limit)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

    return alpha




def choose_color_screen(screen):
    font = pygame.font.SysFont("Arial", 32)
    text_white = font.render("Press W to play as White", True, BLACK)
    text_black = font.render("Press B to play as Black", True, BLACK)

    screen.fill(WHITE)
    screen.blit(text_white, (WIDTH // 2 - text_white.get_width() // 2, HEIGHT // 2 - text_white.get_height()))
    screen.blit(text_black, (WIDTH // 2 - text_black.get_width() // 2, HEIGHT // 2 + text_black.get_height()))

    pygame.display.flip()

    choosing = True
    player_color = None
    while choosing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    player_color = chess.WHITE
                    choosing = False
                elif event.key == pygame.K_b:
                    player_color = chess.BLACK
                    choosing = False
    return player_color


def load_openings(filename="openings.pgn"):

    openings = []
    with open(filename) as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                node = game
                moves = []
                while node.variations:
                    next_node = node.variation(0)
                    moves.append(next_node.move)
                    node = next_node
                openings.append(moves)
            except ValueError as e:
                print(f"Error parsing PGN: {e}")
                continue
    return openings


def get_opening_move(board, openings):
    current_moves = [move.uci() for move in board.move_stack]

    for opening in openings:
        opening_moves = [move.uci() for move in opening]
        if len(opening_moves) >= len(current_moves):
            match = True
            for i in range(len(current_moves)):
                if current_moves[i] != opening_moves[i]:
                    match = False
                    break
            if match and len(opening_moves) > len(current_moves):
                next_move = opening[len(current_moves)]
                if next_move in board.legal_moves:

                    return next_move

    return None

def iterative_deepening(board, max_depth, time_limit, openings=None):
    start_time = time.time()
    best_move = None
    actual_depth = 0

    for depth in range(1, max_depth + 1):
        if (time.time() - start_time) >= time_limit:
            break
        current_best_move = get_best_move(board, depth, openings, start_time, time_limit)
        if (time.time() - start_time) >= time_limit:
            break
        best_move = current_best_move
        actual_depth = depth

    return best_move, actual_depth

def get_best_move(board, depth, openings=None, start_time=None, time_limit=None):
    if openings and board.fullmove_number < 10:  # Use opening moves for the first 10 plies (5 moves per side)
        opening_move = get_opening_move(board, openings)
        if opening_move is not None:
            return opening_move

    best_move = None
    best_value = float('-inf') if board.turn == chess.WHITE else float('inf')
    alpha = float('-inf')
    beta = float('inf')
    legal_moves = list(board.legal_moves)

    for move in legal_moves:
        if start_time and time_limit and (time.time() - start_time) >= time_limit:
            break

        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move  # Immediate checkmate found, execute it
        board_value = minimax(board, depth - 1, alpha, beta, board.turn == chess.BLACK, start_time, time_limit)
        board.pop()
        if start_time and time_limit and (time.time() - start_time) >= time_limit:
            break
        if board.turn == chess.WHITE:
            if board_value > best_value:
                best_value = board_value
                best_move = move
            alpha = max(alpha, board_value)
        else:
            if board_value < best_value:
                best_value = board_value
                best_move = move
            beta = min(beta, board_value)
        if beta <= alpha:
            break
    return best_move


def computer_move(board, max_depth, openings, time_limit):
    best_move, actual_depth = iterative_deepening(board, max_depth, time_limit, openings)
    if best_move is not None:
        board.push(best_move)
    return best_move, actual_depth


STOCKFISH_PATH = "/Users/arianhoush/Downloads/stockfish/stockfish-macos-x86-64"


def computer_move_stockfish(board, time_limit):
    result = engine.play(board, chess.engine.Limit(time=time_limit))
    print(result)
    print(result.move)
    return result.move



def main():
    global positions_analyzed, depth_searched
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess Game")

    #engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    openings = load_openings("openings.pgn")

    while True:
        player_color = choose_color_screen(screen)
        #testing fen for endgame and delivering checkmate
       #test_fen = "4k3/8/8/8/8/4K3/4R3/4R3 b - - 0 1"
        #test_fen = "8/8/8/8/8/6k1/4r3/6K1 w - - 0 1"
        ##mate2_fen = "8/8/8/P7/8/k7/p7/K2n4 w - - 0 1"
      #  mate1_fen = "8/8/8/8/8/1k6/3q4/1K6 w - - 0 1"
       # mate1_2_fen = "4r1k1/7p/6p1/4B3/6N1/6P1/6K1/8 b - - 0 1"
       # board = chess.Board(mate1_2_fen)
        board = chess.Board()
        images = load_images()
        num_moves = 0
        selected_square = None
        game_over = False
        running = True
        player_turn = board.turn == player_color


        bot_evaluation = 0

        while running:
            if not player_turn and not game_over:
                try:
                    positions_analyzed = 0  # Reset positions analyzed counter
                    #uncomment this
                    best_move, depth_searched = computer_move(board, 10, openings, 5)
                    #comment lines 559 and 560  and 564
                    #print(board)
                    #depth_searched = 5
                   # best_move = computer_move_stockfish(board, 5)

                    if best_move is not None:
                        #also  comment this
                        #board.push(best_move)


                        bot_evaluation = evaluate_board(board)
                except Exception as e:
                    print(f"Error during computer move: {e}")
                player_turn = True

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not game_over and player_turn:
                    square = get_square_under_mouse(player_color)
                    if board.turn == player_color:
                        if board.piece_at(square) is not None and board.color_at(square) == board.turn:
                            selected_square = square
                        else:
                            if selected_square is not None:
                                move = chess.Move(selected_square, square)
                                if move in board.legal_moves:
                                    board.push(move)
                                    num_moves += 1
                                    selected_square = None
                                    player_turn = False

                                    if board.is_checkmate():
                                        winner = "White" if board.turn == chess.BLACK else "Black"
                                        game_over = True
                                        play_again = draw_game_over_popup(screen, winner, num_moves)
                                        if play_again:
                                            running = False
                                        else:
                                            game_over = True

                    if not game_over and board.is_checkmate():
                        winner = "Black" if board.turn == chess.WHITE else "White"
                        game_over = True
                        play_again = draw_game_over_popup(screen, winner, num_moves)
                        if play_again:
                            running = False
                        else:
                            game_over = True

            draw_board(screen, player_color)
            draw_pieces(screen, board, images, player_color)
            highlight_squares(screen, board, selected_square, player_color)
            draw_check(screen, board, player_color)
            draw_side_panel(screen, depth_searched, bot_evaluation, positions_analyzed, num_moves)

            pygame.display.flip()

        if not play_again:
            break
   # engine.quit()
    pygame.quit()
  #  sys.exit()


if __name__ == "__main__":
    main()