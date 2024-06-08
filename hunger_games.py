import random

class Participant:
    def __init__(self, name, user_id, image_link=None):
        self.name = name
        self.user_id = user_id
        self.image_link = image_link

    def __repr__(self):
        return self.name

    def interact(self, scenario, others=None):
        if others:
            scenario(self, others)
        else:
            scenario(self)

class HungerGames:
    def __init__(self, participants):
        self.participants = participants
        self.scenarios = [
            self.find_supplies_scenario,
            self.kill_scenario,
            self.trap_scenario,
            self.beast_scenario,
            self.form_alliance_scenario,
            self.betrayal_scenario,
            self.steal_supplies_scenario,
            self.find_water_scenario,
            self.injure_scenario,
            self.fart_scenario
        ]

    def fart_scenario(self, participant):
        print(f"{participant.name} farted.")
        
    def find_supplies_scenario(self, participant):
        print(f"{participant.name} stumbles upon a hidden cache of supplies.")

    def kill_scenario(self, participant, others):
        other = others[0]
        print(f"{participant.name} stabs {other.name}, killing them.")
        self.participants.remove(other)

    def trap_scenario(self, participant):
        print(f"{participant.name} falls into a trap and is eliminated.")
        self.participants.remove(participant)

    def beast_scenario(self, participant):
        print(f"{participant.name} is attacked by a wild beast and is eliminated.")
        self.participants.remove(participant)

    def form_alliance_scenario(self, participant, others):
        other = others[0]
        print(f"{participant.name} forms an alliance with {other.name}. They decide to work together for now.")

    def betrayal_scenario(self, participant, others):
        other = others[0]
        print(f"{participant.name} betrays {other.name} during the night and kills them.")
        self.participants.remove(other)

    def steal_supplies_scenario(self, participant, others):
        other = others[0]
        print(f"{participant.name} sneaks into {other.name}'s camp and steals their supplies.")

    def find_water_scenario(self, participant):
        print(f"{participant.name} finds a source of fresh water and quenches their thirst.")

    def injure_scenario(self, participant):
        print(f"{participant.name} gets injured while trying to climb a tree.")


    def run_round(self):
        print("A new round begins!")
        random.shuffle(self.participants)

        # Create a copy of the participants list to iterate over
        for participant in self.participants[:]:  # Iterate over a copy
            if len(self.participants) <= 1:
                break

            # Check if the participant is still in the game before proceeding
            if participant in self.participants:
                scenario = random.choice(self.scenarios)
                if scenario in [self.kill_scenario, self.form_alliance_scenario, 
                                self.betrayal_scenario, self.steal_supplies_scenario]:
                    # Ensure the 'other' participant is also still in the game:
                    valid_others = [p for p in self.participants if p != participant and p in self.participants]
                    if valid_others: 
                        others = [random.choice(valid_others)]
                        participant.interact(scenario, others)
                else:
                    participant.interact(scenario)

            scenario = random.choice(self.scenarios)
            if scenario in [self.kill_scenario, self.form_alliance_scenario, self.betrayal_scenario, self.steal_supplies_scenario]:
                others = [random.choice([p for p in self.participants if p != participant])]
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


