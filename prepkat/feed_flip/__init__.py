import typer
from pathlib import Path
from typing_extensions import Annotated
from typing import List, Optional
import pyrap.tables as pt
import numpy as np
from rich import print
from rich.progress import track
from numba import njit

FLIP_KEYWORD = "PREPKAT_FEED_FLIP"


def _feed_flip(
    ms_path: Annotated[
        Path,
        typer.Argument(
            help="Path to measurement set.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True
        )
    ],
    columns: Annotated[
        Optional[List[str]],
        typer.Option(
            "--column",
            help=(
                "Column to which the feed flip should be applied. Can be "
                "specified multiple times."
            )
        )] = ["DATA", "WEIGHT_SPECTRUM", "FLAG"]
):
    
    main_table = pt.table(str(ms_path), readonly=False, ack=False)
    n_row = main_table.nrows()

    for col in columns:

        col_keywords = main_table.getcolkeywords(col)

        if col_keywords.get(FLIP_KEYWORD, False):
            print(
                f"[bold yellow] Column {col} has already been flipped - "
                f"skipping.[/bold yellow]")
            continue  # The column has already been flipped.

        exemplar_row = main_table.getcol(col, nrow=1)

        col_shape = (n_row, *exemplar_row.shape[1:])
        col_dtype = exemplar_row.dtype
        
        if col_shape[-1] != 4:
            raise ValueError("Four correlation data required for feed flip.")

        chunk_n_row = 10000
        chunk = np.empty((chunk_n_row, *col_shape[1:]), dtype=col_dtype)

        desc = f"[cyan]Flipping {col}:[/cyan]"

        for start_row in track(range(0, n_row, chunk_n_row), description=desc):
            
            sel = slice(0, min(n_row - start_row, chunk_n_row))

            main_table.getcolnp(
                col,
                chunk[sel],
                startrow=start_row,
                nrow=sel.stop
            )

            apply_flip(chunk[sel])

            main_table.putcol(
                col,
                chunk[sel],
                startrow=start_row,
                nrow=sel.stop
            )

        main_table.putcolkeyword(col, FLIP_KEYWORD, True)

        print(
            f"[bold green]Successfully flipped column: {col}.[/bold green]"
        )

    main_table.close()
    
    # Update FEED subtable.
    feed_table = pt.table(f"{str(ms_path)}::FEED", readonly=False, ack=False)
    receptor_angles = feed_table.getcol("RECEPTOR_ANGLE")
    receptor_angles[:] = 0  # Zero the receptor angles.
    feed_table.putcol("RECEPTOR_ANGLE", receptor_angles)
    feed_table.putcolkeyword("RECEPTOR_ANGLE", FLIP_KEYWORD, True)
    feed_table.close()


@njit(cache=True, nogil=True)
def apply_flip(chunk):

    chunk = chunk.reshape(-1, 4)

    for i in range(chunk.shape[0]):
        ele = chunk[i]
        xx, xy, yx, yy = ele
        ele[0] = yy
        ele[1] = yx
        ele[2] = xy
        ele[3] = xx