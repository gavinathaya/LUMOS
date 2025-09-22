import sys

def progress_bar(step: int, total_steps: int, *, fill_char: str ='━', width: int=40) -> None:
    """
    Print a progress bar to the console.

    Parameters
    ----------
    step : int
        Current step in the process.
    total_steps : int
        Total number of steps in the process.
    fill_char : str, optional
        Character used to fill the progress bar. Default is '━'.
    width : int, optional
        Width of the progress bar in characters. Default is 40.
    
    Returns
    -------
    None
        This function does not return anything. It prints the progress bar to the console.
    """
    # Filler characters: ━ ╍ █
    # Names in order: "U+2501" "U+2505" "U+2588"
    # Edge characters: ╢ ╟
    # Names in order: "U+2562" "U+2560"
    progress = f"{step+1}/{total_steps} {'\033[92m'}╢{fill_char*((step+1)*width//total_steps):<{width}}╟{'\033[0m'}"
    
    # Pad with spaces to clear leftovers
    progress = progress.ljust(7 * width // 4)  # 30 is arbitrary, adjust as needed

    #Line overwrite logic
    if step + 1 == total_steps:
        sys.stdout.write('\r' + progress + '\n')
    else:
        sys.stdout.write('\r' + progress)
    sys.stdout.flush()

    return None