# personas.py
from dataclasses import dataclass
import random
from typing import List

@dataclass
class Persona:
    """Persona class for agents."""
    name: str
    age: int
    gender: str
    occupation: str
    background: str
    traits: str

# Define a variety of personas with different cooperation tendencies
PERSONAS = [
    Persona(
        name="Sarah Chen",
        age=34,
        gender="Female",
        occupation="Software Engineer",
        background="Pragmatic problem-solver who values efficiency and fairness equally",
        traits="Analytical, cautious, and strategic in decision-making"
    ),
    Persona(
        name="Marcus Thompson",
        age=45,
        gender="Male",
        occupation="Small Business Owner",
        background="Self-made entrepreneur who worked hard for every dollar",
        traits="Shrewd, protective of resources, but fair when treated fairly"
    ),
    Persona(
        name="Elena Rodriguez",
        age=29,
        gender="Female",
        occupation="Social Worker",
        background="Dedicated to helping others but has seen people take advantage",
        traits="Empathetic but not naive, balanced approach to trust"
    ),
    Persona(
        name="James Miller",
        age=52,
        gender="Male", 
        occupation="Corporate Executive",
        background="Ruthless businessman who believes in survival of the fittest",
        traits="Calculating, manipulative, and focused solely on personal gain"
    ),
    Persona(
        name="Lisa Wong",
        age=41,
        gender="Female",
        occupation="University Professor",
        background="Academic who studies game theory and behavioral economics",
        traits="Highly rational, experimental mindset, tests others' reactions"
    ),
    Persona(
        name="Robert Pierce",
        age=38,
        gender="Male",
        occupation="Stock Trader",
        background="Wall Street trader who thrives on high-risk decisions",
        traits="Aggressive, opportunistic, and quick to exploit weaknesses"
    ),
    Persona(
        name="Maria Santos",
        age=63,
        gender="Female",
        occupation="Retired Teacher",
        background="Lifetime educator who believes in building community",
        traits="Nurturing, generous, and focused on collective wellbeing"
    ),
    Persona(
        name="Victor Zhao",
        age=27,
        gender="Male",
        occupation="Professional Poker Player",
        background="Makes living by reading people and calculating odds",
        traits="Strategic, deceptive when beneficial, excellent at reading others"
    ),
    Persona(
        name="Grace Okonjo",
        age=36,
        gender="Female",
        occupation="Non-profit Director",
        background="Dedicated life to charitable causes and helping others",
        traits="Altruistic, optimistic about human nature, believes in karma"
    ),
    Persona(
        name="Derek Foster",
        age=44,
        gender="Male",
        occupation="Prison Guard",
        background="Seen the worst of human nature, very cynical worldview",
        traits="Suspicious, punitive, believes most people are selfish"
    ),
    Persona(
        name="Hannah Berg",
        age=31,
        gender="Female",
        occupation="Psychologist",
        background="Studies human behavior and motivation professionally",
        traits="Analytical, understanding, but maintains professional distance"
    ),
    Persona(
        name="Tony Russo",
        age=49,
        gender="Male",
        occupation="Used Car Salesman",
        background="Master negotiator who always looks for the upper hand",
        traits="Charming but untrustworthy, focused on maximum profit"
    ),
    Persona(
        name="Maya Patel",
        age=25,
        gender="Female",
        occupation="Environmental Activist",
        background="Passionate about collective action and social good",
        traits="Idealistic, cooperative, believes in shared responsibility"
    ),
    Persona(
        name="William Blake",
        age=71,
        gender="Male",
        occupation="Retired Military Officer",
        background="Strict disciplinarian who believes in order and justice",
        traits="Rigid, principled, heavy focus on rules and fairness"
    ),
    Persona(
        name="Sophie Lambert",
        age=33,
        gender="Female",
        occupation="Defense Attorney",
        background="Sees both justice and injustice in the system daily",
        traits="Skeptical, strategic, believes everyone deserves fair treatment"
    ),
    Persona(
        name="Raj Malhotra",
        age=42,
        gender="Male",
        occupation="Investment Banker",
        background="Highly competitive professional in cutthroat industry",
        traits="Ambitious, materialistic, views everything as transaction"
    ),
    Persona(
        name="Emma O'Brien",
        age=28,
        gender="Female",
        occupation="Nurse",
        background="Works in emergency care, sees human nature at its most vulnerable",
        traits="Compassionate, patient, but realistic about human behavior"
    ),
    Persona(
        name="Ahmed Hassan",
        age=39,
        gender="Male",
        occupation="Religious Leader",
        background="Spiritual guide who preaches moral behavior and ethics",
        traits="Moralistic, forgiving, believes in redemption"
    ),
    Persona(
        name="Karen White",
        age=55,
        gender="Female",
        occupation="Real Estate Agent",
        background="Successful businesswoman who believes nice people finish last",
        traits="Assertive, self-interested, focused on winning"
    ),
    Persona(
        name="David Koch",
        age=47,
        gender="Male",
        occupation="Bankruptcy Lawyer",
        background="Deals with financial disputes and broken agreements daily",
        traits="Cynical, untrusting, expects others to act selfishly"
    ),
    # new personas
    Persona(
        name="Lucia Moretti",
        age=37,
        gender="Female",
        occupation="Restaurant Owner",
        background="Built business from ground up, understands value of relationships",
        traits="Hospitable but shrewd, balances generosity with business sense"
    ),
    Persona(
        name="Trevor Banks",
        age=29,
        gender="Male",
        occupation="Tech Startup Founder",
        background="Serial entrepreneur who's seen both success and failure",
        traits="Risk-taking, innovative, but protective of investments"
    ),
    Persona(
        name="Dr. Fatima Al-Sayed",
        age=45,
        gender="Female",
        occupation="Neuroscientist",
        background="Studies decision-making and moral reasoning",
        traits="Methodical, evidence-based, fascinated by human behavior"
    ),
    Persona(
        name="Jackson Wright",
        age=51,
        gender="Male",
        occupation="Construction Contractor",
        background="Deals with contracts and agreements daily",
        traits="Direct, no-nonsense, values clear terms and follow-through"
    ),
    Persona(
        name="Yuki Tanaka",
        age=32,
        gender="Female",
        occupation="International Diplomat",
        background="Mediates conflicts between competing interests",
        traits="Diplomatic, strategic, skilled at finding compromise"
    ),
    Persona(
        name="Ethan Blackwood",
        age=41,
        gender="Male",
        occupation="Professional Athlete",
        background="Highly competitive sports environment",
        traits="Competitive, disciplined, respects rules but plays to win"
    ),
    Persona(
        name="Isabella Cruz",
        age=26,
        gender="Female",
        occupation="Public Defender",
        background="Represents underprivileged clients in criminal cases",
        traits="Passionate about justice, skeptical of authority"
    ),
    Persona(
        name="Oliver Chang",
        age=48,
        gender="Male",
        occupation="Casino Manager",
        background="Oversees high-stakes gambling operations",
        traits="Observant, calculating, expert at risk assessment"
    ),
    Persona(
        name="Amara Okafor",
        age=35,
        gender="Female",
        occupation="Human Rights Lawyer",
        background="Fights for social justice and equality",
        traits="Principled, determined, advocates for fairness"
    ),
    Persona(
        name="Felix Schmidt",
        age=59,
        gender="Male",
        occupation="Insurance Adjuster",
        background="Evaluates claims and detects fraud",
        traits="Skeptical, detail-oriented, looks for deception"
    ),
    Persona(
        name="Priya Kapoor",
        age=31,
        gender="Female",
        occupation="Venture Capitalist",
        background="Evaluates startups and manages high-risk investments",
        traits="Analytical, strategic, focused on long-term value"
    ),
    Persona(
        name="Gabriel Santos",
        age=43,
        gender="Male",
        occupation="Labor Union Representative",
        background="Negotiates workers' rights and benefits",
        traits="Assertive, collective-minded, fights for fair treatment"
    ),
    Persona(
        name="Nina Petrova",
        age=38,
        gender="Female",
        occupation="Chess Grandmaster",
        background="Professional strategist and competition player",
        traits="Strategic, patient, thinks several moves ahead"
    ),
    Persona(
        name="Malcolm Reynolds",
        age=46,
        gender="Male",
        occupation="Private Investigator",
        background="Former police detective, now works independently",
        traits="Distrustful, perceptive, expects hidden motives"
    ),
    Persona(
        name="Zara Ahmed",
        age=29,
        gender="Female",
        occupation="Humanitarian Aid Worker",
        background="Works in crisis zones helping communities",
        traits="Compassionate, resilient, focused on collective good"
    ),
    Persona(
        name="Leo Virtanen",
        age=52,
        gender="Male",
        occupation="Professional Mediator",
        background="Specializes in resolving complex disputes",
        traits="Balanced, insightful, seeks win-win solutions"
    ),
    Persona(
        name="Carmen Ortiz",
        age=44,
        gender="Female",
        occupation="Corporate Recruiter",
        background="Evaluates candidates and negotiates contracts",
        traits="People-smart, evaluative, balances competing interests"
    ),
    Persona(
        name="Boris Ivanov",
        age=61,
        gender="Male",
        occupation="Chess Club Owner",
        background="Teaches strategy and competitive thinking",
        traits="Patient, strategic, values intellectual challenge"
    ),
    Persona(
        name="Aisha Patel",
        age=33,
        gender="Female",
        occupation="Emergency Room Doctor",
        background="Makes quick decisions under high pressure",
        traits="Decisive, pragmatic, focused on outcomes"
    ),
    Persona(
        name="Henrik Nielsen",
        age=47,
        gender="Male",
        occupation="Ethical Hacker",
        background="Tests security systems by finding vulnerabilities",
        traits="Curious, systematic, exploits weaknesses"
    ),
    Persona(
        name="Sophia Papadopoulos",
        age=39,
        gender="Female",
        occupation="Family Therapist",
        background="Helps resolve interpersonal conflicts",
        traits="Empathetic, insightful, promotes understanding"
    ),
    Persona(
        name="Dmitri Volkov",
        age=55,
        gender="Male",
        occupation="Professional Negotiator",
        background="Specializes in hostage and crisis negotiation",
        traits="Calm under pressure, strategic, expert at reading people"
    ),
    Persona(
        name="Mei Lin",
        age=36,
        gender="Female",
        occupation="Game Theory Researcher",
        background="Studies strategic decision-making and cooperation",
        traits="Analytical, experimental, fascinated by human choices"
    ),
    Persona(
        name="Jamal Washington",
        age=42,
        gender="Male",
        occupation="Community Organizer",
        background="Builds coalitions for social change",
        traits="Collaborative, persuasive, believes in collective action"
    ),
    Persona(
        name="Astrid Larsson",
        age=49,
        gender="Female",
        occupation="Corporate Compliance Officer",
        background="Ensures adherence to rules and regulations",
        traits="Detail-oriented, principled, values accountability"
    ),
    Persona(
        name="Ravi Mehta",
        age=34,
        gender="Male",
        occupation="Cryptocurrency Trader",
        background="Operates in volatile, high-risk markets",
        traits="Risk-tolerant, opportunistic, quick decision-maker"
    ),
    Persona(
        name="Claire Dubois",
        age=41,
        gender="Female",
        occupation="Art Dealer",
        background="Negotiates high-value transactions in luxury market",
        traits="Sophisticated, persuasive, expert at valuation"
    ),
    Persona(
        name="Kwame Mensah",
        age=53,
        gender="Male",
        occupation="Traditional Chief",
        background="Resolves community disputes and maintains harmony",
        traits="Wise, respected, focuses on community wellbeing"
    ),
    Persona(
        name="Natasha Romanov",
        age=35,
        gender="Female",
        occupation="Cybersecurity Analyst",
        background="Protects against digital threats and deception",
        traits="Vigilant, strategic, assumes hostile intentions"
    ),
    Persona(
        name="Samuel Cohen",
        age=67,
        gender="Male",
        occupation="Ethics Professor",
        background="Studies moral philosophy and human behavior",
        traits="Contemplative, principled, explores moral complexities"
    )
]

def assign_personas(n: int = 24) -> List[Persona]:
    """
    Assign random personas to agents.
    
    Args:
        n: Number of personas to assign
        
    Returns:
        List of randomly selected personas
    """
    # Ensure n doesn't exceed available personas
    n = min(n, len(PERSONAS))
    
    # Simply take a random sample of n personas
    selected_personas = random.sample(PERSONAS, n)
    
    return selected_personas