# ==========================================================
# category.py
# Category Mapping Module
# ==========================================================

import pandas as pd
import numpy as np


# ==========================================================
# CREATE CATEGORY MAPPING
# ==========================================================

def create_category_mapping(labels):

    cluster_names = {}

    for idx, cluster in enumerate(
        sorted(np.unique(labels)),
        start=1
    ):

        cluster_names[
            cluster
        ] = f"Category {idx}"

    return cluster_names


# ==========================================================
# CREATE RESULT DATAFRAME
# ==========================================================

def create_result_dataframe(
    urls,
    labels
):

    cluster_names = (
        create_category_mapping(
            labels
        )
    )

    categories = [
        cluster_names[label]
        for label in labels
    ]

    result = pd.DataFrame({

        "URL": urls,
        "Cluster": labels,
        "Category": categories

    })

    return result


# ==========================================================
# PRINT CATEGORY-WISE OUTPUT
# ==========================================================

def print_category_groups(
    result
):

    print("\nCategory Wise URLs\n")

    categories = (
        result["Category"]
        .unique()
    )

    for category in categories:

        print(
            "\n" + "=" * 60
        )

        print(category)

        print(
            "=" * 60
        )

        urls = result[
            result["Category"]
            == category
        ]["URL"]

        for url in urls:

            print(
                " •",
                url
            )


# ==========================================================
# PRINT CLUSTER STATISTICS
# ==========================================================

def print_cluster_statistics(
    result
):

    print("\nCluster Statistics\n")

    print(
        "Total URLs:",
        len(result)
    )

    print(
        "Total Categories:",
        result["Category"]
        .nunique()
    )

    cluster_sizes = (
        result.groupby(
            "Category"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nCluster Sizes:\n"
    )

    print(cluster_sizes)