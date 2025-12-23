import subprocess
import time
from playwright.sync_api import sync_playwright

# Global variables to maintain persistent resources
p = None
browser = None
page = None


def open_site():
    # Launch Chrome via batch file
    subprocess.Popen(["open_chess.bat"], shell=True)
    # Give Chrome time to start
    time.sleep(3)

    # Initialize persistent Playwright instance and connect to browser
    global p, browser, page
    p = sync_playwright().start()
    browser = p.chromium.connect_over_cdp("http://localhost:9222")
    pages = [pg for ctx in browser.contexts for pg in ctx.pages if "chess.com" in pg.url]
    if not pages:
        raise ValueError("No chess.com page found in the browser.")
    page = pages[0]


# Converts raw HTML piece data into piece-position data
def _convert_piece(piece):
    if piece[6] == 's':
        return ''.join((piece[16:18], piece[13:15]))
    return ''.join((piece[6:8], piece[16:18]))


# Retrieve board data using the persistent page
def get_board_data():
    global page
    board_data = page.evaluate(
        "() => Array.from(document.querySelectorAll('.piece')).map(p => p.className)"
    )

    # Filter strings for only piece and positional data
    board_data = [_convert_piece(a) for a in board_data]
    return board_data

def determine_side():

    return page.evaluate("document.querySelector('wc-chess-board.flipped') === null")

def determine_last_moved():

    return page.evaluate("""highlights = document.querySelectorAll('.highlight');
    if(highlights.length !== 0){
        e1 = highlights[0];
        e2 = highlights[1];
    
        sq1 = "." + e1.classList[1];
        sq2 = "." + e2.classList[1];
    
        l1 = document.querySelectorAll(sq1);
        if(l1.length === 2){
            if(l1[1].classList[1][0] !== 's'){
                l1[1].classList[1][0];
            }
            else if(l1[0].classList[1][0] !== 's'){
                l1[0].classList[1][0];
            }
        }        
        else{
            l2 = document.querySelectorAll(sq2);
            if(l2.length === 2){
                if(l2[1].classList[1][0] !== 's'){
                l2[1].classList[1][0];
            }
            else{
                l2[0].classList[1][0];
            }
            }
        }
    }""")
