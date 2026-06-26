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
    import torch
    import torchvision
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    import matplotlib.pyplot as plt

    return T, mo, plt, torch, torchvision


@app.cell
def _(mo):
    # Options for dataset
    # TODO: Add local ImageNet support (like the paper)
    resolution = mo.ui.slider(start=32, stop=256, step=32, value=128, label="Resolution")

    mo.md(f"### Dataset Settings\n{resolution}")
    return (resolution,)


@app.cell
def _(T, mo, resolution, torchvision):
    # Import and transform dataset
    transform = T.Compose([
        T.Resize(resolution.value),
        T.CenterCrop(resolution.value),
        T.ToTensor(),
    ])

    try:
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        categories = dataset.classes
        data_ready = True

    except Exception as e:
        data_ready = False
        error_msg = str(e)

    if not data_ready:
        mo.stop(True, mo.callout(f"Error loading dataset: {error_msg}", kind="danger"))
    return categories, dataset


@app.cell
def _(categories, mo):
    category_selection = mo.ui.dropdown(
        options=categories,
        value=categories[0],
        label="Select Category"
    )

    num_samples = mo.ui.number(
        start=4, stop=16, step=4, value=8, label="Number of Samples"
    )

    mo.hstack([category_selection, num_samples])
    return category_selection, num_samples


@app.cell
def _(
    categories,
    category_selection,
    dataset,
    mo,
    num_samples,
    plt,
    torch,
    torchvision,
):
    target_label = categories.index(category_selection.value)
    images = []

    for img, label in dataset:
        if label == target_label:
            images.append(img)
        if len(images) >= num_samples.value:
            break

    # Image grid
    if images:
        grid = torchvision.utils.make_grid(torch.stack(images), nrow=4, padding=2)

        plt.figure(figsize=(10, 5))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f"Samples for category: {category_selection.value}")

        result = mo.as_html(plt.gca().get_figure())
    else:
        result = mo.md("No images found for this category.")

    result
    return


if __name__ == "__main__":
    app.run()
