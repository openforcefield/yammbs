import abc

from openff.models.models import DefaultModel
from openmm import System
from pydantic import validator

from ib.forcefields import GAFFForceFieldProvider, SMIRNOFFForceFieldProvider
from ib.molecules import OpenFFSingleMoleculeTopologyProvider


class SystemProvider(DefaultModel, abc.ABC):
    identifier: str

    @classmethod
    @abc.abstractmethod
    def allowed_sources(cls):
        raise NotImplementedError()

    @validator("*")
    def validate_sources(cls, value):
        if type(value) in cls.allowed_sources():
            return value
        else:
            raise ValueError(f"Unsupported type {type(value)}")

    @abc.abstractmethod
    def to_system(self):
        raise NotImplementedError()


class SMIRNOFFSystemProvider(SystemProvider):
    identifier: str = "smirnoff"
    topology: OpenFFSingleMoleculeTopologyProvider
    force_field: SMIRNOFFForceFieldProvider

    @classmethod
    def allowed_sources(cls):
        return [
            OpenFFSingleMoleculeTopologyProvider,
            SMIRNOFFForceFieldProvider,
        ]

    def to_system(self) -> System:
        return self.force_field.to_object().create_openmm_system(
            self.topology.to_topology()
        )


class GAFFSystemProvider(SystemProvider):
    identifier: str = "gaff"
    topology: OpenFFSingleMoleculeTopologyProvider
    force_field: GAFFForceFieldProvider

    @classmethod
    def allowed_sources(cls):
        return [
            OpenFFSingleMoleculeTopologyProvider,
            GAFFForceFieldProvider,
        ]

    def to_system(self) -> System:
        return self.force_field.to_object().createSystem(
            self.topology.to_topology().to_openmm(),
        )
