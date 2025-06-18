### The Milvus Vector Database RESTful API

This is a Python based RESTful API for testing the functionality of Milvus Vector Database using FastAPI.

To get the inital knowledge and config the DB, follow the Medium Article Series on <a href="https://medium.com/@malindumadhubashana/a-beginners-guide-to-milvus-vector-database-part-i-2e84a11a29d2">Here</a>

### Example Request for ANN Search

```/api/Vector/annsearch```

```
{
  "query_vector": [0.35, -0.60, 0.18, 0.22],
  "top_k": 3,
  "collection_name": "iris_data",
  "field_name": "features"
}
```
### Example Request for ANN Filtered Search

```/api/Vector/annfilteredsearch```

```
{
  "query_vector": [0.35, -0.60, 0.18, 0.22],
  "top_k": 6,
  "collection_name": "iris_data",
  "field_name": "features",
  "filter": "species like \"Iris%\"",
  "output_fields": ["species"]
}
```
### Example Request for Hybrid Search

```/api/Vector/hybridsearch```

```
{
  "query_vectors": [
    [0.35, -0.60, 0.18, 0.22],
    [0.45, -0.50, 0.20, 0.25]
  ],
  "weights": [0.6, 0.4],
  "collection_name": "iris_data",
  "vector_field": "features",
  "per_query_limit": 5,
  "combined_limit": 10,
  "filter": "species like \"Iris%\"",
  "output_fields": ["id", "species"],
  "metric_type": "L2",
  "consistency_level": "Strong"
}
```
