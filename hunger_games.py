import random

items = [
    "a first-aid kit", "a hunting knife", "a water purifier", "a rope", 
    "a map of the arena", "a backpack full of food", "a set of camouflage clothing", 
    "a compass", "a torch", "a bow and arrow", "a sturdy axe", "a metal pan", 
    "a bottle of water", "a whistle", "a magnifying glass", "a fishing rod", 
    "a thermal blanket", "a sturdy shovel", "a set of waterproof matches"
]

class Participant:
    def __init__(self, name, user_id, avatar=None):
        self.name = name
        self.user_id = user_id
        self.avatar = avatar
        self.alive = True
        self.allies = []

    def __repr__(self):
        return self.name

    def interact(self, scenario, others=None):
        if others:
            scenario(self, others)
        else:
            scenario(self)

    def form_alliance(self, other):
        if other not in self.allies:
            self.allies.append(other)
            other.allies.append(self)

    def betray(self, other):
        if other in self.allies:
            self.allies.remove(other)
            other.allies.remove(self)

class HungerGames:
    def __init__(self, participants):
        self.participants = participants
        self.scenarios = [
            self.find_supplies_scenario, self.kill_scenario, self.trap_scenario, self.beast_scenario,
            self.form_alliance_scenario, self.betrayal_scenario, self.steal_supplies_scenario,
            self.find_water_scenario, self.injure_scenario, self.fart_scenario, self.poop_scenario,
            self.item_kill_scenario, self.eat_berries_scenario, self.tripping_scenario,
            self.sleeping_scenario, self.sponsor_scenario, self.tree_sleep_scenario, self.haunt_scenario,
            self.fire_scenario, self.help_scenario
        ]

    def fire_scenario(self, participant):
        print(f"{participant} causes a fire causing others to run from the area")

    def haunt_scenario(self, participant):
        print(f"{participant} is haunted by nightmares and wakes up in a panic.")

    def tree_sleep_scenario(self, participant):
        print(f"{participant.name} slept in a tree for shelter")

    def tripping_scenario(self, participant):
        trip_item = random.choice(items)
        print(f"{participant.name} tripped on {trip_item} and got injured.")

    def help_scenario(self, participant, others):
        other = others[0]
        print(f"{participant} helped {other}")

    def beast_scenario(self, participant):
        print(f"{participant.name} is attacked by a wild beast and is eliminated.")
        self.participants.remove(participant)

    def fart_scenario(self, participant):
        print(f"{participant.name} farted.")

    def eat_berries_scenario(self, participant):
        if random.random() < 0.5:
            print(f"{participant.name} found some berries and ate them. They were delicious!")
        else:
            print(f"{participant.name} found some berries and ate them. Unfortunately, they were poisonous! {participant.name} is eliminated.")
            self.participants.remove(participant)

    def poop_scenario(self, participant):
        print(f"{participant.name} pooped too hard.")
        self.participants.remove(participant)

    def item_kill_scenario(self, participant, others):
        other = others[0]
        murder_item = random.choice(items)
        print(f"{participant.name} kills {other.name} with {murder_item}.")
        self.participants.remove(other)

    def find_supplies_scenario(self, participant):
        found_item = random.choice(items)
        print(f"{participant.name} stumbles upon {found_item}.")

    def kill_scenario(self, participant, others):
        other = others[0]
        print(f"{participant.name} stabs {other.name}, killing them.")
        self.participants.remove(other)

    def trap_scenario(self, participant):
        print(f"{participant.name} falls into a trap and is eliminated.")
        self.participants.remove(participant)

    def form_alliance_scenario(self, participant, others):
        other = others[0]
        participant.form_alliance(other)
        print(f"{participant.name} forms an alliance with {other.name}. They decide to work together for now.")

    def betrayal_scenario(self, participant, others):
        other = others[0]
        participant.betray(other)
        print(f"{participant.name} betrays {other.name} during the night and kills them.")
        self.participants.remove(other)


    def steal_supplies_scenario(self, participant, others):
        other = others[0]
        participant.betray(other)
        print(f"{participant.name} sneaks into {other.name}'s camp and steals their supplies.")

    def find_water_scenario(self, participant):
        print(f"{participant.name} finds a source of fresh water and quenches their thirst.")

    def injure_scenario(self, participant):
        print(f"{participant.name} gets injured while trying to climb a tree.")

    def sponsor_scenario(self, participant):
        sponsor_item = random.choice(items)
        print(f"{participant.name} was given {sponsor_item} by an unknown sponsor")

    def sleeping_scenario(self, participant, others):
        other = others[0]
        participant.form_alliance(other)
        print(f"{participant.name} slept with {other.name}.")

    def choose_scenario(self, participant):
        weights = []
        for scenario in self.scenarios:
            if scenario in [self.form_alliance_scenario, self.help_scenario, self.sleeping_scenario]:
                weight = 1 if participant.allies else 5
            elif scenario in [self.betrayal_scenario, self.kill_scenario, self.item_kill_scenario, self.steal_supplies_scenario]:
                weight = 1 if any(other in participant.allies for other in self.participants) else 5
            else:
                weight = 1
            weights.append(weight)
        
        return random.choices(self.scenarios, weights=weights, k=1)[0]


    def run_event(self):
        participant = random.choice(self.participants)
        scenario = self.choose_scenario(participant)
        
        if scenario in [self.kill_scenario, self.item_kill_scenario, self.betrayal_scenario, self.form_alliance_scenario, self.steal_supplies_scenario, self.sleeping_scenario]:
            others = random.sample([p for p in self.participants if p != participant], k=1)
            participant.interact(scenario, others)
        else:
            participant.interact(scenario)

    def simulate(self):
        round_number = 1
        while len(self.participants) > 1:
            print(f"=== Round {round_number} ===")
            self.run_round()
            round_number += 1

        print("The game has ended!")
        if self.participants:
            print(f"The winner is {self.participants[0]}")
        else:
            print("No one survived.")
