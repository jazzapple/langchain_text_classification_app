from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from rich.console import Console
from rich.table import Table

CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

console = Console()


def run_evaluation(chain, dataset, label_map: dict):
    predictions = []
    true_labels = []

    console.print(f"\n[bold]Classifying {len(dataset)} articles...[/bold]\n")

    for i, item in enumerate(dataset):
        result = chain.invoke({"text": item["text"]})
        predictions.append(result.category)
        true_labels.append(label_map[item["label"]])

        console.print(
            f"[dim]{i + 1:3}.[/dim] "
            f"True: [cyan]{label_map[item['label']]:10}[/cyan] "
            f"Pred: [{'green' if result.category == label_map[item['label']] else 'red'}]{result.category:10}[/]  "
            f"[dim]{result.reasoning[:80]}[/dim]"
        )

    acc = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions, labels=CATEGORIES)

    _print_confusion_matrix(cm)
    _print_per_class_metrics(true_labels, predictions)
    console.print(f"\n[bold green]Accuracy: {acc:.1%}[/bold green] ({sum(p == t for p, t in zip(predictions, true_labels))}/{len(dataset)} correct)\n")


def _print_per_class_metrics(true_labels, predictions):
    report = classification_report(true_labels, predictions, labels=CATEGORIES, output_dict=True)

    table = Table(title="\nPer-Class Metrics")
    table.add_column("Category", style="bold")
    table.add_column("Precision", justify="center")
    table.add_column("Recall", justify="center")
    table.add_column("F1 Score", justify="center")
    table.add_column("Support", justify="center")

    for cat in CATEGORIES:
        m = report[cat]
        table.add_row(
            cat,
            f"{m['precision']:.2f}",
            f"{m['recall']:.2f}",
            f"{m['f1-score']:.2f}",
            str(int(m["support"])),
        )

    console.print(table)


def _print_confusion_matrix(cm):
    table = Table(title="\nConfusion Matrix (rows=true, cols=predicted)")
    table.add_column("True \\ Pred", style="bold")
    for cat in CATEGORIES:
        table.add_column(cat, justify="center")

    for i, cat in enumerate(CATEGORIES):
        table.add_row(cat, *[str(cm[i][j]) for j in range(len(CATEGORIES))])

    console.print(table)
