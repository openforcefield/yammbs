import abc
import json
from typing import List

from openff.models.models import DefaultModel
from openff.toolkit import Molecule, Topology
from pydantic import conlist


class TopologyComponent(DefaultModel, abc.ABC):
    identifier: str

    @abc.abstractmethod
    def to_json(self):
        raise NotImplementedError()


class OpenFFMolecule(TopologyComponent):
    identifier = "openff"
    molecule: Molecule

    @classmethod
    def allowed_components(cls):
        return [Molecule]

    def to_json(self):
        return json.dumps(
            {
                "identifier": self.identifier,
                "molecule": json.loads(self.molecule.to_json()),
            }
        )


class TopologyProvider(DefaultModel, abc.ABC):
    identifier: str
    components: List[TopologyComponent]
    is_periodic: bool

    @classmethod
    @abc.abstractmethod
    def allowed_components(cls):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_json(self):
        raise NotImplementedError()


class OpenFFSingleMoleculeTopologyProvider(TopologyProvider):
    identifier = "openff"
    components: conlist(OpenFFMolecule, min_items=1, max_items=1)
    is_periodic = False

    @classmethod
    def allowed_components(cls):
        return [OpenFFMolecule]

    def to_topology(self) -> Topology:
        return Topology.from_molecules([self.components[0].molecule])

    def to_json(self):
        return self.to_topology().to_json()
