import json
from pathlib import Path

from .base import AbstractTenantStore, TenantConfig


class LocalTenantStore(AbstractTenantStore):
    """JSON file-based tenant store for dev / single-node deployments."""

    def __init__(self, path: str | Path = "data/tenants.json"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, tenant_id: str) -> TenantConfig:
        store = self._load()
        if tenant_id not in store:
            raise KeyError(tenant_id)
        return self._from_dict(tenant_id, store[tenant_id])

    def save(self, cfg: TenantConfig) -> None:
        store = self._load()
        store[cfg.tenant_id] = self._to_dict(cfg)
        self._path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_by_key(self, api_key: str) -> TenantConfig | None:
        for tid, data in self._load().items():
            if data.get("api_key") == api_key:
                return self._from_dict(tid, data)
        return None

    def list_all(self) -> list[TenantConfig]:
        return [self._from_dict(tid, d) for tid, d in self._load().items()]

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    @staticmethod
    def _from_dict(tenant_id: str, d: dict) -> TenantConfig:
        return TenantConfig(
            tenant_id=tenant_id,
            name=d.get("name", tenant_id),
            plan=d.get("plan", "basic"),
            active=d.get("active", True),
            api_key=d.get("api_key"),
            meta={k: v for k, v in d.items()
                  if k not in ("name", "plan", "active", "api_key")},
        )

    @staticmethod
    def _to_dict(cfg: TenantConfig) -> dict:
        # Standard keys come last so they cannot be overridden by meta.
        return {
            **cfg.meta,
            "name":    cfg.name,
            "plan":    cfg.plan,
            "active":  cfg.active,
            "api_key": cfg.api_key,
        }
