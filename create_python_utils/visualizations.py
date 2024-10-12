import matplotlib.pyplot as plt
import seaborn as sns

def confusionMatrixHeatMap(cm):
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Negative", "Predicted Positive"],
        yticklabels=["Actual Negative", "Actual Positive"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def featureDistributions(df):
    df_melted = df.melt(var_name="Feature", value_name="Value")
    g = sns.FacetGrid(
        df_melted, col="Feature", col_wrap=4, height=3, sharex=False, sharey=False
    )
    g.map(sns.histplot, "Value", bins=20, edgecolor="k")
    g.set_titles("{col_name}")
    g.set_axis_labels("Value", "Frequency")
    for ax in g.axes.flat:
        if ax.get_title() == "work_type" or ax.get_title() == "smoking_status":
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")
    plt.tight_layout()
    plt.show()

def feature_residual(feature, residuals):
    feature = feature.astype(str)

    plt.figure(figsize=(10, 8))
    plt.scatter(feature, residuals, color="blue")
    plt.xlabel(feature.name.capitalize())
    plt.ylabel("Residuals")
    plt.title(f"{feature.name.capitalize()} vs. Residuals")
    plt.xticks(rotation=45, ha="right")
    plt.show()


def actual_pred(pred, actual):
    plt.scatter(actual, pred, color="blue", label="Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs. Predicted")
    plt.show()


def residual_pred(pred, residual):
    plt.scatter(pred, residual)
    plt.axhline(0, color="r", linestyle="--")
    plt.title("Residuals vs Predicted Values")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()

def plot_correlation_heatmap(df):
    # Compute the correlation matrix
    corr = df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        annot=False,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, left=0.25)
    plt.show()

def plot_outliers(df):
    for col in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()

