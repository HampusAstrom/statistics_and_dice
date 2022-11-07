import numpy as np
import numpy.random as rng
from numpy.random import choice
from operator import itemgetter

MAX_PARENT_VALUE = 3
SKILLS_DICT = {
    "Physics": ["Astronomy", "Nuclear"],
    "Biology": ["Exobiology", "Terrestrial"],
    "Chemistry": ["Synthesis", "Analysis"],
    "Psycology": ["Social", "Clinical"],
    "Introspection": ["Capability", "Condition"],
    "Martial": ["Hand-2-Hand", "Firearms"],
    "Engineering": ["Electrical", "Structural"],
    "Medicine": ["Pathology", "Cryo and Trauma"],
    "Computers": ["Hardware", "Software"],
    "Piloting": ["Orbital", "Atmospheric"],
    "Athletics": ["Gravitational", "Zero-G"],
    "Communication": ["Leadership", "Personal"]
}

BASE_SIZE = len(SKILLS_DICT)
IGNORE_FACTOR = 0.1
MAIN_FACTOR = 1.
CORE_FACTOR = 2.


class skill:
    """

    """

    def __init__(self, name, parent=None, children=None):
        self.name = name
        self.parent = parent
        self.value = 0
        self.children = {}
        if children:
            self.add_children(children)

    def __repr__(self):
        if self.parent:
            return f"Skill({self.name}, {self.parent.name}, {self.value}, {self.children})"
        return f"Skill({self.name}, None, {self.value}, {self.children})"

    def add_child(self, child):
        ch = skill(child, self)
        self.children[child] = ch

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def get_total_value(self):
        if self.parent:
            return self.parent.get_total_value() + self.value
        else:
            return self.value

    def increase_value(self, add = 1):
        if not isinstance(add, int) or add < 1:
            raise  ValueError("adding less than 1")
        if self.parent and self.parent.value is not MAX_PARENT_VALUE:
            #print("Warning, parent {} not fully increased, adding value there first".format(self.parent))
            val = min(MAX_PARENT_VALUE - self.parent.value, add)
            self.parent.increase_value(val)
            add -= val
            if add < 1:
                return None, 0
        if self.children:
            if self.value == MAX_PARENT_VALUE:
                return self.children, add
            elif self.value + add > MAX_PARENT_VALUE:
                self.value = MAX_PARENT_VALUE
                return self.children, add - (MAX_PARENT_VALUE - self.value)
        self.value += add
        return None, 0


    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def get_tree(self, start=True):
        if start and self.parent is not None:
            return self.parent.get_tree()

        ret = [(self.name, self.get_total_value())]

        if self.children:
            for _, child in self.children.items():
                ret.append(child.get_tree(start=False))
        return ret

class character:

    def __init__(self):
        self.skills = self.prep_skills()

    def prep_skills(self):
        skills = {}
        for parent, children in SKILLS_DICT.items():
            skills[parent] = skill(parent, children=children)
        return skills

    def get_skills(self):
        return self.skills

    def get_skill(self, name, parent):
        if parent:
            return self.skills[parent].children[name]
        return self.skills[name]

    def __str__(self):
        # TODO: add other field when they exist
        s = "Skills: \n"
        t = ""
        for skill in self.skills.values():
            s += f"{skill.name:<15} {skill.value} -> {list(skill.children)[0]:<15} {list(skill.children.values())[0].value} {list(skill.children.values())[0].value + skill.value}\n"
            s += f"{t:<17} `> {list(skill.children)[1]:<15} {list(skill.children.values())[1].value} {list(skill.children.values())[1].value + skill.value} \n"
        return s


class department_generator:

    def __init__(self, dep_name, core, main, ignore, commanders, spread_factor):
        self.department_name = dep_name
        self.st = self.gen_skill_table(core, main, ignore, commanders)
        self.core = core
        self.main = main
        self.ignore = ignore
        self.commanders = commanders
        self.spread_factor = spread_factor

    def check_input(self, skilltable, input, input_name):
        for s in input:
            if not isinstance(s, str):
                raise TypeError("Error, {} in {} is not a string and is not used!".format(s, input_name))
            if s not in skilltable:
                print("Warning, {} in {} is not a listed skill and is not used!".format(s, input_name))


    def gen_skill_table(self, core, main, ignore, commanders):
        """
        dict of all subskills for char gen with each entry having fields:
        skill, parent, base_prio, individual_prio, department_point_sum
        """
        skill_table = {}
        for parent in SKILLS_DICT:
            skill_table[parent] = [parent, None, 1., 1., 0]
            for child in SKILLS_DICT[parent]:
                skill_table[child] = [child, parent, 0.2, 0.2, 0]

        self.check_input(skill_table, core, "core")
        self.check_input(skill_table, main, "main")
        self.check_input(skill_table, ignore, "ignore")
        self.check_input(skill_table, commanders, "commanders")
        for skill in commanders:
            if skill in main or skill in core:
                print("Warning, a commander skill is also present on the main or core list! Can lead to unexpected behaviour")

        for skill in ignore:
            skill_table[skill][2] = IGNORE_FACTOR
            skill_table[skill][3] = IGNORE_FACTOR
        for skill in main:
            skill_table[skill][2] = BASE_SIZE*MAIN_FACTOR
            skill_table[skill][3] = BASE_SIZE*MAIN_FACTOR
        for skill in commanders: # lowered proportionally after commanders are made
            skill_table[skill][2] = BASE_SIZE*MAIN_FACTOR
            skill_table[skill][3] = BASE_SIZE*MAIN_FACTOR
        for skill in core:
            skill_table[skill][2] = BASE_SIZE*CORE_FACTOR
            skill_table[skill][3] = BASE_SIZE*CORE_FACTOR

        return skill_table

    def gen_characters(self, num_chars, min_points, mean_points, name_gen=None):
        def calc_prio_sum():
            s = 0
            for skill in self.st.values():
                s += skill[3]
            return s

        def points_to_add(char_points):
            if char_points >= mean_points:
                return min(rng.randint(3, 5), char_points)
            if char_points >= min_points:
                return min(rng.randint(2, 4), char_points)
            if char_points >= min_points*0.5:
                return min(rng.randint(1, 4), char_points)
            return min(rng.randint(1, 2), char_points)

        def add_to_skill(p, skill):
            children, remainder = skill.increase_value(p)
            if children:
                entries = itemgetter(*children.keys())(self.st)
                weights = np.array([entry[3] for entry in entries])
                if sum(weights) <= 0:
                    weights += 1
                names = [entry[0] for entry in entries]
                c = choice(names, p=weights/sum(weights))
                add_to_skill(remainder, skill.children[c])

        if mean_points < min_points or not isinstance(num_chars, int):
            raise ValueError("gen_characters input requirements violated")

        d = np.random.default_rng().dirichlet(np.ones(num_chars),size=1)
        budget = np.rint(min_points + d * (mean_points-min_points) * num_chars)[0].astype(np.int32)
        budget = sorted(budget, reverse=True)
        n_commanders = int(np.ceil(num_chars/10))
        print(budget)
        print(n_commanders)

        weight_sum = calc_prio_sum()
        commandes_active = True

        for i, char_points in enumerate(budget):
            char = character()
            char_sum = weight_sum
            while char_points > 0:
                weights = np.array([entry[3] for entry in self.st.values()])
                try:
                    skill_name = choice(list(self.st.keys()), p=weights/char_sum)
                except ValueError as err:
                    print(err)
                p = points_to_add(char_points)
                add_to_skill(p, char.get_skill(skill_name, self.st[skill_name][1]))
                char_points -= p
                self.st[skill_name][4] += p # not completely accurate, as point might have gone to parent
                char_sum -= self.st[skill_name][3]
                self.st[skill_name][3] *= 0.2
                char_sum += self.st[skill_name][3]

                t = self.st[skill_name][2]
                new_weight = t * (1 - self.spread_factor/num_chars)
                weight_sum -= (t - new_weight)
                self.st[skill_name][2] = new_weight
            # check for and remove commander prio if appropriate
            if commandes_active and i > n_commanders:
                for skill_name in self.commanders:
                    red = self.st[skill_name][2]/MAIN_FACTOR
                    if skill_name in self.ignore:
                        self.st[skill_name][2] = red*IGNORE_FACTOR
                    else:
                        self.st[skill_name][2] = red*1.
            # reset individual_prio
            for skill in self.st.values():
                skill[3] = skill[2]
            print(char)
        for skill in self.st.values():
            print(f"{skill[0]} {skill[4]}")






if __name__ == "__main__":
    """
    rudimentary testing
    """
    #s1 = skill("Physics", children=SKILLS_DICT["Physics"])
    #s1.get_children()["Nuclear"].increase_value(3)
    #print(s1.get_total_value())
    #print(s1.get_tree())

    #d = department_generator("test", ["Nuclear", "Engineering"], ["Physics", "Introspection", "Computers", "Chemistry"], ["Martial", "Biology"], ["Communication", "Introspection"], 0.1)
    #d.gen_characters(10, 15, 20)
    d = department_generator("test", ["Engineering"], ["Hardware", "Orbital", "Leadership"], ["Biology", "Medicine", "Psycology"], [], 0.1)
    #d = department_generator("test", [], ["Physics", "Cryo and Trauma", "Chemistry", "Psycology", "Introspection"], ["Piloting", "Martial", ], ["Leadership"], 0.1)
    d.gen_characters(20, 10, 16)
