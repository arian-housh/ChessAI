import pygame
import chess
import sys

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# Colors
WHITE = pygame.Color("antiquewhite")
BLACK = pygame.Color("chocolate4")
BLUE = pygame.Color("blue")
GREEN = pygame.Color("green")
RED = pygame.Color("red")

# Load images
def load_images():
    pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
    images = {}
    for piece in pieces:
        images[piece] = pygame.image.load(f'ChessPieces/{piece}.png')
    return images

# Draw the board
def draw_board(screen):
    colors = [WHITE, BLACK]
    for row in range(ROWS):
        for col in range(COLS):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

# Draw the pieces
def draw_pieces(screen, board, images):
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_symbol = piece.symbol()
        piece_color = 'w' if piece.color == chess.WHITE else 'b'
        piece_key = f'{piece_color}{piece_symbol.upper()}'
        screen.blit(images[piece_key], (col * SQUARE_SIZE, row * SQUARE_SIZE))

# Highlight squares
def highlight_squares(screen, board, selected_square):
    if selected_square is not None:
        row, col = divmod(selected_square, 8)
        pygame.draw.rect(screen, BLUE, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)
        for move in board.legal_moves:
            if move.from_square == selected_square:
                to_row, to_col = divmod(move.to_square, 8)
                pygame.draw.rect(screen, GREEN, pygame.Rect(to_col * SQUARE_SIZE, to_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

def get_square_under_mouse():
    x, y = pygame.mouse.get_pos()
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return chess.square(col, row)

def draw_check(screen, board):
    if board.is_check():
        king_square = board.king(board.turn)
        row, col = divmod(king_square, 8)
        pygame.draw.rect(screen, RED, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

def draw_game_over_popup(screen, winner, num_moves):
    popup_width, popup_height = 300, 200
    popup_x, popup_y = (WIDTH - popup_width) // 2, (HEIGHT - popup_height) // 2
    font = pygame.font.SysFont("Arial", 24)

    # Draw popup background
    pygame.draw.rect(screen, WHITE, (popup_x, popup_y, popup_width, popup_height))
    pygame.draw.rect(screen, BLACK, (popup_x, popup_y, popup_width, popup_height), 2)

    # Draw text
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

    # Wait for user input
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    return True  # Play again
                elif event.key == pygame.K_n:
                    return False  # Do not play again

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess")

    board = chess.Board()
    images = load_images()
    num_moves = 0

    selected_square = None
    game_over = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                square = get_square_under_mouse()
                if board.piece_at(square) is not None and board.color_at(square) == board.turn:
                    selected_square = square
                else:
                    if selected_square is not None:
                        piece = board.piece_at(selected_square)
                        move = chess.Move(selected_square, square)
                        if move in board.legal_moves:
                            board.push(move)
                            num_moves += 1

                            if board.is_checkmate():
                                winner = "White" if board.turn == chess.BLACK else "Black"
                                game_over = True
                                play_again = draw_game_over_popup(screen, winner, num_moves)
                                if play_again:
                                    board.reset()
                                    num_moves = 0
                                    game_over = False
                                else:
                                    selected_square = None  # Allow examining the final position
                        else:
                            selected_square = None  # Clear the selection if the move is illegal

        draw_board(screen)
        draw_pieces(screen, board, images)
        highlight_squares(screen, board, selected_square)
        draw_check(screen, board)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
