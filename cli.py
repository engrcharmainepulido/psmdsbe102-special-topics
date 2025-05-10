import typer
from tune import run_hyperparameter_tuning
from evaluate import evaluate_best_model
import uvicorn

app = typer.Typer()

@app.command()
def train():
    """Run hyperparameter tuning and save best model."""
    typer.echo("Running training...")
    run_hyperparameter_tuning("heart_disease_results")

@app.command()
def evaluate():
    """Evaluate the best model and generate confusion matrix."""
    typer.echo("Evaluating model...")
    evaluate_best_model()

@app.command()
def serve():
    """Start the FastAPI prediction server."""
    typer.echo("Starting server at http://127.0.0.1:8000")
    uvicorn.run("serve:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    app()
