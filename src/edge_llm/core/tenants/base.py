import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TenantConfig:
    tenant_id: str
    name: str = ""
    plan: str = "basic"
    active: bool = True
    api_key: str | None = None
    # Domain-specific config — each vertical adds its own fields
    # by subclassing TenantConfig or passing extra as `meta`.
    meta: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        """Convenience accessor for meta fields."""
        return self.meta.get(key, default)


class AbstractTenantStore(ABC):

    @abstractmethod
    def get(self, tenant_id: str) -> TenantConfig:
        """Return config for tenant_id. Raises KeyError if not found."""

    @abstractmethod
    def save(self, cfg: TenantConfig) -> None:
        """Persist or update a tenant config."""

    @abstractmethod
    def get_by_key(self, api_key: str) -> TenantConfig | None:
        """Look up a tenant by its API key. Returns None if not found."""

    @abstractmethod
    def list_all(self) -> list[TenantConfig]:
        """Return all registered tenants."""

    def register(self, tenant_id: str, name: str = "", plan: str = "basic",
                 meta: dict | None = None) -> str:
        """
        Create a new tenant, generate an API key, persist and return the key.
        Raises ValueError if tenant_id already exists.
        """
        try:
            self.get(tenant_id)
            raise ValueError(f"Tenant '{tenant_id}' already exists.")
        except KeyError:
            pass
        api_key = "sk-" + secrets.token_hex(24)
        self.save(TenantConfig(
            tenant_id=tenant_id,
            name=name or tenant_id,
            plan=plan,
            active=True,
            api_key=api_key,
            meta=meta or {},
        ))
        return api_key
