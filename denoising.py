import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Back to Basics: Let Denoising Generative Models Denoise
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Documentation:
    - Paper: https://www.alphaxiv.org/abs/2511.13720
    - Video review of the paper: https://www.youtube.com/watch?v=u5yKZzTTEHo
    - Traditional technic used: https://www.youtube.com/watch?v=iv-5mZ_9CPY
    """)
    return


if __name__ == "__main__":
    app.run()
