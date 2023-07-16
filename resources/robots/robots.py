#!/usr/bin/env python
"""Module for robots

Module that stores base robot abstract class :class:`BaseRobot` from which all
the other agent classes must to inherit. 

This class is implemented in such a way that adding new robots is as easy as
possible. 

For creating new robot user simply needs:
    * :attr:`source_file` - Path to the robot XML config file.
    * :attr:`picture_path` - Path to a picture of robot/placeholder.
    * :attr:`environment_id` - ID of Farama environment to be used (env used for custom robots ``"custom/CustomEnv-v0"``)

Default implemented robots:
    #. Custom robot/env
        * :class:`StickAnt`
        * :class:`AntV3`
        * :class:`SpotLike`
    #. Farama NEAT robots
        * :class:`Walker2D`
        * :class:`InvertedPendulum`
        * :class:`InvertedDoublePendulum`
"""

from abc import ABC, abstractproperty, abstractclassmethod
import os
import re
import copy
import tempfile
import numpy as np

DIR = os.path.dirname(__file__)

class BaseRobot(ABC):
    """Base robot abstract class.

    This is the base class for all robots from which they must inherit and call
    its :func:`__init__` constructor with parameters described above (:attr:`source_file`,
    :attr:`picture_path`, :attr:`environment_id`).

    The :func:`__init__` stores all needed information and parses the robot's XML
    config file for possible changable body parts.

    .. note:: 
        Body parts are specified inside robot's XML config file with
        specialised symbols ``$...$`` and ``@...@``. More about those symbols
        in :ref:`body_parts_symbols`.

    Every robot class stores following variables.

    :ivar source_file: Path to robot's XML source file (or None for NEAT only robots)
    :ivar picture_path: Path to a picture of the robot or a placeholder.
    :ivar environment_id: ID of environment (example: ``"custom/CustomEnv-v0"`` or ``"Walker2d-v4"``)
    :ivar body_parts: Dictionary of found body parts in XML (= parsed variable symbols)

    :vartype source_file: str, Optional
    :vartype picture_path: str
    :vartype environment_id: str
    :vartype body_parts: Dict[str, float]
    """

    def __init__(self, source_file, picture_path, env_id):
        self.environment_id = env_id # defined in custom evn script
        self.picture_path = picture_path

        self.body_parts = {}

        self.source_file = source_file
        if source_file != None:
            with open(source_file) as file:
                lines = file.readlines()
                self.source_text = "".join(lines)

            self.__collect_body_parts()

    @abstractproperty
    def description(self):
        """Robot description.

        Informative property used in GUI robot selection, presenting
        information on robot. 

        Returns:
            str : Informative text.
        """
        pass

    def create(self, body_part_mask, individual=([],[]), tmp_file=None):
        """
        Creates/writes XML file (in place) - robot specific XML string with
        changed body part variables. 

        Robot's XML source file may include special symbols which we parse
        and evaluate (symbols for body parts ``$...$``, symbols for arithmetic
        evaluation ``@...@``). This method gets body part mask intended to show
        which body parts can be changed and which should stay at default
        values, individual (possibly empty for *default* robots) and possibly
        file (if one is already created for this purpose - reuse) to which is
        the parsed XML config going to be written.

        Method uses specific regular expressions to find special symbols.
        Method also uses :attr:`body_parts` dictionary created on init by
        :func:`__collect_body_parts` method (also using robot's XML).

        Args:
            body_part_mask (List[Tuple[float]|False]) : Body part mask
                including for each body part either range of allowed values or
                False.
            individual (Tuple[List[...],List[...]]) : Individual from EA -
                containing actions in first item of tuple and adjustmetns for
                body part lengths in second item.
            tmp_file (TemporaryFileWrapper[str]|None) : Parameter used when
                reusing already created temporary files for this purpose.
                Parsed XML config is rewritten inside of the passed file (and
                still returned).

        Returns:
            temporary file : Returns temporary file with parsed XML config (or 
            none for robots without source file - robots for NEAT) 
        """

        def key_to_regex(key):
            regex = key.replace("$", "\$")
            regex = regex.replace("(", "\(")
            regex = regex.replace(")", "\)")

            return regex

        if self.source_file == None:
            return None

        if tmp_file == None:
            # TMP files needs to stay existing after closing (windows does not
            # allow to use already opened files)
            tmp_file = tempfile.NamedTemporaryFile(mode="w",suffix=".xml",prefix="GArobot_",delete=False)
        else:
            tmp_file = open(tmp_file.name, "w")

        _, body_part_adjustments = individual

        text = copy.deepcopy(self.source_text)
        
        # adjust body part variables
        idx = 0
        for i, key in enumerate(self.body_parts):
            regex = key_to_regex(key)
            # set to desired body part length if True in mask ELSE set to a default value given by xml source file
            if body_part_mask[i]:
                text = re.sub(regex, str(body_part_adjustments.flatten()[idx]), text)
                idx += 1
            else:
                text = re.sub(regex, str(self.body_parts[key]), text)

        # compute body parts calculations
        calculations = re.findall(r'@.*@', text)
        for calc in calculations:
            _calc = calc.strip('@')
            text = re.sub(calc, str(eval(_calc)), text)

        # clear and change file in place
        tmp_file.seek(0)
        tmp_file.truncate()
        tmp_file.write(text)
        tmp_file.flush()

        return tmp_file

    def create_default(self):
        """
        Method which uses the :func:`create` method to create default of
        selected robot with default body part lengths (if any are included).
        Used when initialising :class:`gaAgents.BaseAgent` to created simple
        test environment to get the input and observation sizes.
        """
        tmp_file = self.create(body_part_mask=np.zeros([len(self.body_parts)]))

        return tmp_file

    def __collect_body_parts(self):
        """
        Uses regular expressions to parse names and default lengths of all
        changable body parts from XML source file.
        """

        part_names = re.findall(r'\$[A-Za-z0-9_]+\([+-]?[0-9]*[.]?[0-9]+\)\$', self.source_text)
        for part in part_names:
            if not part in self.body_parts:
                self.body_parts[part] = float(part[part.find("(")+1 : part.find(")")]) # get between parenthesis

    @property
    def body_part_names(self):
        """
        Property for listing found changable body parts - used in GUI for body
        part unlocking.
        """

        return list(self.body_parts.keys())

class StickAnt(BaseRobot):
    def __init__(self):
        source_file = DIR+"/assets/custom_stick_ant.xml"
        picture_path = DIR+"/assets/Basic-Ant"
        environment_id = "custom/CustomEnv-v0"

        super(StickAnt, self).__init__(source_file, picture_path, environment_id)

    @property
    def description(self):
        return "The simplest robot similar to the original Ant, missing the \
last part of its legs - legs are using only single Z direction hinge joint. It \
is the simplest default robot in this library with only 4 degrees of freedom. \
Same as the Ant, the length of each leg part can also be evolved independantly." 

class AntV3(BaseRobot):
    def __init__(self):
        source_file = DIR+"/assets/ant.xml"
        picture_path = DIR+"/assets/Ant-v3"
        environment_id = "custom/CustomEnv-v0"

        super(AntV3, self).__init__(source_file, picture_path, environment_id)

    @property
    def description(self):
        return "Copy of the original MuJoCo Ant. Its body is made of a single \
sphere with 4 legs, each having 2 parts. Each leg has 2 joints - 1 hip \
joint (hinge joint in Z direction) and 1 knee joint (hinge joint parallel \
to the ground). The robot is augmented with changable body parts - the length \
of each leg part can be separately evolved during morphology evolution. \
The robot inside the original and our custom environment is tested for flip \
over which is deemed as unhealthy position - causes termination of the \
simulation."

class SpotLike(BaseRobot):
    def __init__(self):
        source_file = DIR+"/assets/spot_like.xml"
        picture_path = DIR+"/assets/SpotLike"
        environment_id = "custom/CustomEnv-v0"

        super(SpotLike, self).__init__(source_file, picture_path, environment_id)

    @property
    def description(self):
        return "Homage to BostonDynamics and their great dog like robot Spot \
used as an inspiration when modeling our custom SpotLike robot. \
The robot has 4 legs made of 2 parts each (thigh and calf ended with fixed foot). \
All together the robot has 12 joints (12 degrees of freedomo) - 2 hinge joints \
for each hip free to rotate along X and Y axis and 1 hinge joint for each knee \
along Y axis. This robot can have 3 possible body parts enabled for evolution: \n\
    * THIGH_X - how far back will the thigh go alongside the body. \n\
    * THIGH_Z - how low will the thigh go (height of knee). \n\
    * CALF_X  - how far forward will the calf go (opposite direction to THIGH_X). \
"

class Walker2D(BaseRobot):
    def __init__(self):
        source_file = None
        picture_path = DIR+"/assets/Walker2D"
        environment_id = "Walker2d-v4"

        super(Walker2D, self).__init__(source_file, picture_path, environment_id)

    @property
    def description(self):
        return "The 2D walker robot from the MuJoCo environment. It is a \
2D two-legged figure with a single torso and thigh, calf and foot for \
each leg. The robot is controlled with 6 hinge joints (3 for each leg): \n\
    * thigh_joint \n\
    * leg_joint \n\
    * foot_joint \n\
This robot is good for testing NEAT agents."

class InvertedDoublePendulum(BaseRobot):
    def __init__(self):
        source_file = None
        picture_path = DIR+"/assets/double_invertedPendulum"
        environment_id = "InvertedDoublePendulum-v4"

        super(InvertedDoublePendulum, self).__init__(source_file, picture_path, environment_id)

    @property
    def description(self):
        return "A classical InvertedDoublePendulum robot improving on the \
cartpole environment idea with second pole on top of the first one. The \
cart can move only on single axis. The goal is to balance the second pole \
on top of the first one. Only useful for NEAT experiments."

class InvertedPendulum(BaseRobot):
    def __init__(self):
        source_file = None
        picture_path = DIR+"/assets/invertedPendulum"
        environment_id = "InvertedPendulum-v4"

        super(InvertedPendulum, self).__init__(source_file, picture_path, environment_id)

    @property
    def description(self):
        return "Easier version of InvertedDoublePendulum, using only single \
pole on top of the cart. With much more predictable motion the balancing goal \
is pretty easy to satisfy - might be useful to implement other penalties (for \
example distance from the center - even on long sims, the carts tend to drift \
slowly to the edge)."
