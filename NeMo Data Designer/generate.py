# generate_reviews.py
import os
from dotenv import load_dotenv

from nemo_microservices.data_designer.essentials import (
    CategorySamplerParams,
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
    NeMoDataDesignerClient,
    PersonSamplerParams,
    SamplerColumnConfig,
    SamplerType,
    SubcategorySamplerParams,
    UniformSamplerParams,
)

# --- Load .env ---
load_dotenv()
API_KEY = os.getenv("NEMO_DD_API_KEY")
if not API_KEY:
    raise RuntimeError("NEMO_DD_API_KEY is not set in your .env file")

# --- Client ---
data_designer_client = NeMoDataDesignerClient(
    base_url="https://ai.api.nvidia.com/v1/nemo/dd",
    default_headers={"Authorization": f"Bearer {API_KEY}"},
)

# Choose a hosted trial model alias
model_alias = "nemotron-nano-v2"

# --- Build dataset config ---
config_builder = DataDesignerConfigBuilder()

config_builder.add_column(
    SamplerColumnConfig(
        name="product_category",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["Electronics", "Clothing", "Home & Kitchen", "Books", "Home Office"]
        ),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="product_subcategory",
        sampler_type=SamplerType.SUBCATEGORY,
        params=SubcategorySamplerParams(
            category="product_category",
            values={
                "Electronics": ["Smartphones", "Laptops", "Headphones", "Cameras", "Accessories"],
                "Clothing": ["Men's Clothing", "Women's Clothing", "Winter Coats", "Activewear", "Accessories"],
                "Home & Kitchen": ["Appliances", "Cookware", "Furniture", "Decor", "Organization"],
                "Books": ["Fiction", "Non-Fiction", "Self-Help", "Textbooks", "Classics"],
                "Home Office": ["Desks", "Chairs", "Storage", "Office Supplies", "Lighting"],
            },
        ),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="target_age_range",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(values=["18-25", "25-35", "35-50", "50-65", "65+"]),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="customer",
        sampler_type=SamplerType.PERSON,
        params=PersonSamplerParams(age_range=[18, 70]),
    )
)

# Use high=6 with convert_to=int to ensure 1..5 range in some SDKs
config_builder.add_column(
    SamplerColumnConfig(
        name="number_of_stars",
        sampler_type=SamplerType.UNIFORM,
        params=UniformSamplerParams(low=1, high=6),
        convert_to="int",
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="review_style",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["rambling", "brief", "detailed", "structured with bullet points"],
            weights=[1, 2, 2, 1],
        ),
    )
)

# LLM columns (order matters: product_name before customer_review)
config_builder.add_column(
    LLMTextColumnConfig(
        name="product_name",
        prompt=(
            "Come up with a creative product name for a product in the '{{ product_category }}' category, focusing "
            "on products related to '{{ product_subcategory }}'. The target age range of the ideal customer is "
            "{{ target_age_range }} years old. Respond with only the product name, no other text."
        ),
        system_prompt=(
            "You are a helpful assistant that generates product names. You respond with only the product name, "
            "no other text. You do NOT add quotes around the product name."
        ),
        model_alias=model_alias,
    )
)

config_builder.add_column(
    LLMTextColumnConfig(
        name="customer_review",
        prompt=(
            "You are a customer named {{ customer.first_name }} from {{ customer.city }}, {{ customer.state }}. "
            "You are {{ customer.age }} years old and recently purchased a product called {{ product_name }}. "
            "Write a review of this product, which you gave a rating of {{ number_of_stars }} stars. "
            "The style of the review should be '{{ review_style }}'."
        ),
        model_alias=model_alias,
    )
)

# --- Generate 10 records ---
preview = data_designer_client.preview(config_builder, num_records=10)

# Show one sample record in console
preview.display_sample_record()

# Get DataFrame and print first rows
df = preview.dataset
print(df.head(3))

# Optionally save to CSV
df.to_csv("product_reviews.csv", index=False)
print("Saved product_reviews.csv")