from memarena.cache import IngestionCache, ingestion_cache_key
from memarena.providers.base import ProviderInfo


def _info(name="p", client_version="1.0", config_digest="abc"):
    return ProviderInfo(name=name, client_version=client_version, config_digest=config_digest,
                         pricing_model="per_token")


class TestIngestionCacheKey:
    def test_key_differs_by_namespace(self):
        k1 = ingestion_cache_key(_info(), dataset_digest="d1", namespace="ns1")
        k2 = ingestion_cache_key(_info(), dataset_digest="d1", namespace="ns2")
        assert k1 != k2

    def test_key_differs_by_provider_name(self):
        k1 = ingestion_cache_key(_info(name="mem0"), dataset_digest="d1", namespace="ns1")
        k2 = ingestion_cache_key(_info(name="baseline_rag"), dataset_digest="d1", namespace="ns1")
        assert k1 != k2

    def test_key_differs_by_client_version(self):
        k1 = ingestion_cache_key(_info(client_version="1.0"), dataset_digest="d1", namespace="ns1")
        k2 = ingestion_cache_key(_info(client_version="2.0"), dataset_digest="d1", namespace="ns1")
        assert k1 != k2

    def test_key_differs_by_config_digest(self):
        k1 = ingestion_cache_key(_info(config_digest="abc"), dataset_digest="d1", namespace="ns1")
        k2 = ingestion_cache_key(_info(config_digest="xyz"), dataset_digest="d1", namespace="ns1")
        assert k1 != k2

    def test_key_differs_by_dataset_digest(self):
        k1 = ingestion_cache_key(_info(), dataset_digest="d1", namespace="ns1")
        k2 = ingestion_cache_key(_info(), dataset_digest="d2", namespace="ns1")
        assert k1 != k2

    def test_key_is_stable(self):
        k1 = ingestion_cache_key(_info(), dataset_digest="d1", namespace="ns1")
        k2 = ingestion_cache_key(_info(), dataset_digest="d1", namespace="ns1")
        assert k1 == k2


class TestIngestionCache:
    def test_unseen_key_is_not_ingested(self):
        cache = IngestionCache()
        assert cache.already_ingested("k1") is False

    def test_marked_key_is_ingested(self):
        cache = IngestionCache()
        cache.mark_ingested("k1")
        assert cache.already_ingested("k1") is True

    def test_marking_is_isolated_per_instance(self):
        cache_a, cache_b = IngestionCache(), IngestionCache()
        cache_a.mark_ingested("k1")
        assert cache_b.already_ingested("k1") is False
