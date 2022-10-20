import abc

from openff.models.models import DefaultModel
from openff.units import Quantity
from openmm import System
from openmm.app import Topology

from ib.forcefields import GAFFForceFieldProvider, SMIRNOFFForceFieldProvider
from ib.molecules import OpenFFSingleMoleculeTopologyProvider


class SystemProvider(DefaultModel, abc.ABC):
    identifier: str

    @classmethod
    @abc.abstractmethod
    def allowed_sources(cls):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_system(self):
        raise NotImplementedError()


class SMIRNOFFSystemProvider(SystemProvider):
    identifier: str = "smirnoff"
    topology: OpenFFSingleMoleculeTopologyProvider
    force_field: SMIRNOFFForceFieldProvider
    positions: Quantity

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

    def to_openmm_topology(self) -> Topology:
        return self.topology.to_topology().to_openmm()


class GAFFSystemProvider(SystemProvider):
    identifier: str = "gaff"
    topology: OpenFFSingleMoleculeTopologyProvider
    force_field: GAFFForceFieldProvider
    positions: Quantity

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

    def to_openmm_topology(self) -> Topology:
        return self.topology.to_topology().to_openmm()
