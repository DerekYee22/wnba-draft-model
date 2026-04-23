// Default team archetype need scores (0–10 scale, higher = greater need)
// These are placeholder values — run the WNBA needs model notebook and
// export team_needs.json to /public/ to override with actual model output.
//
// Format per team:
//   archetypeNeeds: { [archetypeName]: score }
//   topStatNeeds: string[] (stat names the team is most deficient in)

export const DEFAULT_TEAM_NEEDS = {
  'Atlanta Dream': {
    archetypeNeeds: {
      'Floor General':    7,
      'Post Scorer':    6,
      'Combo Guard':      5,
      '3-and-D Wing':     5,
      'Stretch Big':      7,
      'Interior Big':     8,
    },
    topStatNeeds: ['Interior Defense', 'Rebounding', 'Scoring'],
    notes: 'Needs a frontcourt anchor and a capable shot creator.',
  },
  'Chicago Sky': {
    archetypeNeeds: {
      'Floor General':    8,
      'Post Scorer':    9,
      'Combo Guard':      7,
      '3-and-D Wing':     6,
      'Stretch Big':      6,
      'Interior Big':     7,
    },
    topStatNeeds: ['Scoring', 'Playmaking', 'Rebounding'],
    notes: 'Rebuilding — need impact at every position.',
  },
  'Connecticut Sun': {
    archetypeNeeds: {
      'Floor General':    7,
      'Post Scorer':    8,
      'Combo Guard':      6,
      '3-and-D Wing':     7,
      'Stretch Big':      5,
      'Interior Big':     5,
    },
    topStatNeeds: ['Wing Scoring', 'Shot Creation', '3-Point Shooting'],
    notes: 'Strong defensive core, needs offensive creation on the wing.',
  },
  'Dallas Wings': {
    archetypeNeeds: {
      'Floor General':    7,
      'Post Scorer':    8,
      'Combo Guard':      7,
      '3-and-D Wing':     8,
      'Stretch Big':      6,
      'Interior Big':     6,
    },
    topStatNeeds: ['Scoring', 'Defense', 'Versatility'],
    notes: 'Rebuilding roster needs high-impact talent at all positions.',
  },
  'Golden State Valkyries': {
    archetypeNeeds: {
      'Floor General':    9,
      'Post Scorer':    9,
      'Combo Guard':      8,
      '3-and-D Wing':     8,
      'Stretch Big':      8,
      'Interior Big':     9,
    },
    topStatNeeds: ['Scoring', 'Rebounding', 'Defense', 'Playmaking'],
    notes: 'Expansion team — building from scratch, all archetypes in demand.',
  },
  'Indiana Fever': {
    archetypeNeeds: {
      'Floor General':    4,
      'Post Scorer':    5,
      'Combo Guard':      6,
      '3-and-D Wing':     7,
      'Stretch Big':      7,
      'Interior Big':     8,
    },
    topStatNeeds: ['Interior Defense', 'Wing Depth', 'Rebounding'],
    notes: 'Have Caitlin Clark as playmaker — need frontcourt and wing support.',
  },
  'Las Vegas Aces': {
    archetypeNeeds: {
      'Floor General':    5,
      'Post Scorer':    6,
      'Combo Guard':      7,
      '3-and-D Wing':     8,
      'Stretch Big':      6,
      'Interior Big':     5,
    },
    topStatNeeds: ['Wing Depth', '3-Point Shooting', 'Support Pieces'],
    notes: 'Championship core intact — looking for high-ceiling depth.',
  },
  'Los Angeles Sparks': {
    archetypeNeeds: {
      'Floor General':    9,
      'Post Scorer':    9,
      'Combo Guard':      7,
      '3-and-D Wing':     6,
      'Stretch Big':      6,
      'Interior Big':     7,
    },
    topStatNeeds: ['Scoring', 'Playmaking', 'Interior Defense'],
    notes: 'Rebuilding; need a franchise cornerstone creator.',
  },
  'Minnesota Lynx': {
    archetypeNeeds: {
      'Floor General':    6,
      'Post Scorer':    7,
      'Combo Guard':      6,
      '3-and-D Wing':     8,
      'Stretch Big':      5,
      'Interior Big':     5,
    },
    topStatNeeds: ['Wing Scoring', '3-Point Shooting', 'Depth'],
    notes: 'Experienced core — looking for wing scoring and versatility.',
  },
  'New York Liberty': {
    archetypeNeeds: {
      'Floor General':    4,
      'Post Scorer':    5,
      'Combo Guard':      7,
      '3-and-D Wing':     8,
      'Stretch Big':      6,
      'Interior Big':     6,
    },
    topStatNeeds: ['Depth', 'Wing Defense', 'Rebounding'],
    notes: 'Championship contender looking for high-value role players.',
  },
  'Phoenix Mercury': {
    archetypeNeeds: {
      'Floor General':    7,
      'Post Scorer':    8,
      'Combo Guard':      6,
      '3-and-D Wing':     5,
      'Stretch Big':      6,
      'Interior Big':     7,
    },
    topStatNeeds: ['Scoring', 'Interior Defense', 'Rebounding'],
    notes: 'Need scoring alongside Brittney Griner and a frontcourt partner.',
  },
  'Seattle Storm': {
    archetypeNeeds: {
      'Floor General':    7,
      'Post Scorer':    8,
      'Combo Guard':      7,
      '3-and-D Wing':     6,
      'Stretch Big':      6,
      'Interior Big':     6,
    },
    topStatNeeds: ['Shot Creation', 'Scoring', 'Wing Versatility'],
    notes: 'Transitioning — need a new offensive centerpiece.',
  },
  'Washington Mystics': {
    archetypeNeeds: {
      'Floor General':    9,
      'Post Scorer':    9,
      'Combo Guard':      8,
      '3-and-D Wing':     7,
      'Stretch Big':      7,
      'Interior Big':     8,
    },
    topStatNeeds: ['Scoring', 'Defense', 'Rebounding', 'Playmaking'],
    notes: 'Full rebuild — all archetype needs are elevated.',
  },
  'Toronto Tempo': {
    archetypeNeeds: {
      'Floor General':    8,
      'Post Scorer':      9,
      'Combo Guard':      8,
      '3-and-D Wing':     8,
      'Stretch Big':      7,
      'Interior Big':     9,
    },
    topStatNeeds: ['Scoring', 'Playmaking', 'Rim Protection', 'Rebounding'],
    notes: 'Inaugural season — building from scratch, all archetypes in demand.',
  },
  'Portland Fire': {
    archetypeNeeds: {
      'Floor General':    8,
      'Post Scorer':      9,
      'Combo Guard':      8,
      '3-and-D Wing':     8,
      'Stretch Big':      7,
      'Interior Big':     9,
    },
    topStatNeeds: ['Scoring', 'Playmaking', 'Rim Protection', 'Rebounding'],
    notes: 'Inaugural season — building from scratch, all archetypes in demand.',
  },
}
