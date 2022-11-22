import abc
from typing import List

from openff.models.models import DefaultModel
from openff.toolkit import ForceField, Molecule
from openmm.app import ForceField as OpenMMForceField


class ForceFieldProvider(DefaultModel, abc.ABC):
    identifier: str

    @classmethod
    @abc.abstractmethod
    def allowed_sources(cls):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_object(self):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_identifier(cls):
        raise NotImplementedError()


class SMIRNOFFForceFieldProvider(ForceFieldProvider):
    identifier: str = "smirnoff"
    # This could just be a str ... trust that it will be a well-formed
    # input and parse it into an object according to some assumptions ...
    force_field: ForceField

    @classmethod
    def allowed_sources(cls) -> List:
        return [ForceField]

    def to_object(self):
        return self.force_field

    @classmethod
    def from_identifier(cls, identifier: str):
        force_field_name = identifier
        if not force_field_name.endswith(".offxml"):
            force_field_name += ".offxml"

        return cls(
            identifier=identifier,
            force_field=ForceField(force_field_name),
        )


class GAFFForceFieldProvider(ForceFieldProvider):
    identifier: str = "gaff"
    force_field: OpenMMForceField

    @classmethod
    def from_molecule(cls, molecule: Molecule, gaff_version: str = "2.11"):
        from openmmforcefields.generators import GAFFTemplateGenerator

        gaff_generator = GAFFTemplateGenerator(molecules=molecule).generator

        force_field = OpenMMForceField()
        force_field.registerTemplateGenerator(gaff_generator)

        return cls(
            identifier=gaff_version,
            force_field=force_field,
        )

    @classmethod
    def allowed_sources(cls) -> List:
        return [OpenMMForceField]

    def to_object(self):
        return self.force_field

    @classmethod
    def from_identifier(cls, identifier: str, molecule: Molecule):
        if not identifier.startswith("gaff"):
            raise ValueError("Only GAFF force fields are supported")

        from openmmforcefields.generators import GAFFTemplateGenerator

        gaff_generator = GAFFTemplateGenerator(
            molecules=molecule, forcefield=identifier
        )
        gaff_forcefield = OpenMMForceField()
        gaff_forcefield.registerTemplateGenerator(gaff_generator.generator)
        return cls(identifier=identifier, force_field=gaff_forcefield)
