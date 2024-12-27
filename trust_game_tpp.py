from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass
import asyncio

@dataclass
class GameState:
    """Represents the current state of an interaction"""
    helper_endowment: float = 10.0
    recipient_endowment: float = 0.0
    punishment_cost: float = 2.0
    punishment_impact: float = 6.0
    helping_amount: float = 5.0

@dataclass
class InteractionResult:
    """Records the results of a single interaction"""
    helper_helped: bool
    punisher_punished: bool
    helper_final_payoff: float
    recipient_final_payoff: float
    punisher_final_payoff: float

class Agent:
    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
        self.interaction_history: List[InteractionResult] = []
        self.public_reputation = 0.0
        
    async def get_decision(self, state: GameState, context: str) -> bool:
        """Base method for agent decisions. Override in specific agent types."""
        raise NotImplementedError

class Helper(Agent):
    async def get_decision(self, state: GameState, context: str) -> bool:
        """Decide whether to help recipient"""
        prompt = f"""You are {self.name}, deciding whether to share money with a recipient in an economic game.

Current situation:
- You have ${state.helper_endowment}
- You can choose to give ${state.helping_amount} to the recipient
- This is a {context} interaction

Previous interactions visible to others: {self.interaction_history if 'public' in context else 'Not visible'}

Think step by step:
1. Consider the potential benefits and costs
2. Consider your reputation {' and how this may affect future interactions' if 'public' in context else ''}
3. Make your decision

Would you share the money with the recipient? Respond with only 'True' or 'False'."""

        # This would call the LLM with the prompt
        # For now, simulate with a probability
        return random.random() < 0.6

class Punisher(Agent):
    async def get_decision(self, state: GameState, context: str, helper_decision: bool) -> bool:
        """Decide whether to punish the helper"""
        if helper_decision:  # No punishment needed if helper cooperated
            return False
            
        prompt = f"""You are {self.name}, deciding whether to punish a selfish action in an economic game.

Current situation:
- Another player chose not to share ${state.helping_amount} with a recipient
- You can pay ${state.punishment_cost} to reduce their payoff by ${state.punishment_impact}
- This is a {context} interaction
- Your decision {'will' if 'public' in context else 'will not'} be visible to future interaction partners

Previous interactions visible to others: {self.interaction_history if 'public' in context else 'Not visible'}

Think step by step:
1. Consider whether the helper's action deserves punishment
2. Evaluate the cost to you versus the impact on the helper
3. Consider {'how this might signal your character to others' if 'public' in context else 'your moral principles'}
4. Make your decision

Would you punish the selfish action? Respond with only 'True' or 'False'."""

        # This would call the LLM with the prompt
        # For now, simulate with different probabilities for public vs anonymous
        return random.random() < (0.7 if 'public' in context else 0.4)

class Chooser(Agent):
    async def decide_trust_amount(self, state: GameState, partner_history: List[InteractionResult]) -> float:
        """Decide how much to trust the partner in the trust game"""
        visible_history = [result for result in partner_history if hasattr(result, 'public')]
        
        prompt = f"""You are {self.name}, deciding how much money to send to a partner in a trust game.

Current situation:
- You have ${state.helper_endowment} to potentially share
- Any amount you send will be tripled
- Your partner can then decide how much to return to you
- Partner's previous behavior: {visible_history}

Think step by step:
1. Evaluate partner's past behavior and trustworthiness
2. Consider the potential return on investment
3. Decide on an amount that balances trust and risk
4. Make your decision

How much would you send (0-10)? Respond with only a number."""

        # This would call the LLM with the prompt
        # For now, simulate based on partner's punishment history
        punishment_rate = sum(1 for x in visible_history if x.punisher_punished) / max(len(visible_history), 1)
        base_trust = 0.5  # Base trust level
        trust_modifier = punishment_rate * 0.3  # Increased trust for punishers
        return state.helper_endowment * (base_trust + trust_modifier)

class TPPExperiment:
    def __init__(self):
        self.helpers = [Helper(f"Helper_{i}", "helper") for i in range(5)]
        self.punishers = [Punisher(f"Punisher_{i}", "punisher") for i in range(5)]
        self.choosers = [Chooser(f"Chooser_{i}", "chooser") for i in range(5)]
        
    async def run_interaction(self, context: str) -> InteractionResult:
        """Run a single TPP interaction"""
        state = GameState()
        
        # Randomly select agents
        helper = random.choice(self.helpers)
        punisher = random.choice(self.punishers)
        
        # Get decisions
        helped = await helper.get_decision(state, context)
        punished = await punisher.get_decision(state, context, helped) if not helped else False
        
        # Calculate payoffs
        helper_payoff = state.helper_endowment
        if helped:
            helper_payoff -= state.helping_amount
        elif punished:
            helper_payoff -= state.punishment_impact
            
        recipient_payoff = state.helping_amount if helped else 0
        punisher_payoff = -state.punishment_cost if punished else 0
        
        return InteractionResult(helped, punished, helper_payoff, recipient_payoff, punisher_payoff)

    async def run_experiment(self, n_rounds: int, context: str):
        """Run multiple rounds of the experiment"""
        results = []
        for _ in range(n_rounds):
            result = await self.run_interaction(context)
            results.append(result)
            
        # Calculate statistics
        punishment_rate = sum(1 for r in results if r.punisher_punished) / n_rounds
        helping_rate = sum(1 for r in results if r.helper_helped) / n_rounds
        
        return {
            'punishment_rate': punishment_rate,
            'helping_rate': helping_rate,
            'results': results
        }

async def main():
    # Run experiments in both public and anonymous conditions
    experiment = TPPExperiment()
    
    public_results = await experiment.run_experiment(100, "public")
    anonymous_results = await experiment.run_experiment(100, "anonymous")
    
    print(f"Public condition:")
    print(f"Punishment rate: {public_results['punishment_rate']:.2%}")
    print(f"Helping rate: {public_results['helping_rate']:.2%}")
    
    print(f"\nAnonymous condition:")
    print(f"Punishment rate: {anonymous_results['punishment_rate']:.2%}")
    print(f"Helping rate: {anonymous_results['helping_rate']:.2%}")

if __name__ == "__main__":
    asyncio.run(main())