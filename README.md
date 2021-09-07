# shinra-pipeline

# Quick start using Docker
```bash
docker build . --tag shinra
docker run -it -v $PWD/src:/workspace -v /path/to/data_path:/src -v /path/to/model_path:/models shinra
```

# extract attributes
```bash
mkdir models
cd attribute_extraction
sh predict.sh
```