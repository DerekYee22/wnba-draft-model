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
      'Primary Creator': 7,
      'Interior Defender': 8,
      'Balanced Contributor': 6,
      'Support Player': 4,
    },
    topStatNeeds: ['Interior Defense', 'Rebounding', 'Scoring'],
    notes: 'Needs a frontcourt anchor and a capable shot creator.',
  },
  'Chicago Sky': {
    archetypeNeeds: {
      'Primary Creator': 9,
      'Interior Defender': 7,
      'Balanced Contributor': 8,
      'Support Player': 5,
    },
    topStatNeeds: ['Scoring', 'Playmaking', 'Rebounding'],
    notes: 'Rebuilding — need impact at every position.',
  },
  'Connecticut Sun': {
    archetypeNeeds: {
      'Primary Creator': 8,
      'Interior Defender': 5,
      'Balanced Contributor': 7,
      'Support Player': 6,
    },
    topStatNeeds: ['Wing Scoring', 'Shot Creation', '3-Point Shooting'],
    notes: 'Strong defensive core, needs offensive creation on the wing.',
  },
  'Dallas Wings': {
    archetypeNeeds: {
      'Primary Creator': 8,
      'Interior Defender': 6,
      'Balanced Contributor': 9,
      'Support Player': 7,
    },
    topStatNeeds: ['Scoring', 'Defense', 'Versatility'],
    notes: 'Rebuilding roster needs high-impact talent at all positions.',
  },
  'Golden State Valkyries': {
    archetypeNeeds: {
      'Primary Creator': 9,
      'Interior Defender': 9,
      'Balanced Contributor': 9,
      'Support Player': 8,
    },
    topStatNeeds: ['Scoring', 'Rebounding', 'Defense', 'Playmaking'],
    notes: 'Expansion team — building from scratch, all archetypes in demand.',
  },
  'Indiana Fever': {
    archetypeNeeds: {
      'Primary Creator': 5,
      'Interior Defender': 8,
      'Balanced Contributor': 7,
      'Support Player': 6,
    },
    topStatNeeds: ['Interior Defense', 'Wing Depth', 'Rebounding'],
    notes: 'Have Caitlin Clark as playmaker — need frontcourt and wing support.',
  },
  'Las Vegas Aces': {
    archetypeNeeds: {
      'Primary Creator': 6,
      'Interior Defender': 5,
      'Balanced Contributor': 7,
      'Support Player': 8,
    },
    topStatNeeds: ['Wing Depth', '3-Point Shooting', 'Support Pieces'],
    notes: 'Championship core intact — looking for high-ceiling depth.',
  },
  'Los Angeles Sparks': {
    archetypeNeeds: {
      'Primary Creator': 9,
      'Interior Defender': 7,
      'Balanced Contributor': 8,
      'Support Player': 6,
    },
    topStatNeeds: ['Scoring', 'Playmaking', 'Interior Defense'],
    notes: 'Rebuilding; need a franchise cornerstone creator.',
  },
  'Minnesota Lynx': {
    archetypeNeeds: {
      'Primary Creator': 7,
      'Interior Defender': 5,
      'Balanced Contributor': 6,
      'Support Player': 7,
    },
    topStatNeeds: ['Wing Scoring', '3-Point Shooting', 'Depth'],
    notes: 'Experienced core — looking for wing scoring and versatility.',
  },
  'New York Liberty': {
    archetypeNeeds: {
      'Primary Creator': 5,
      'Interior Defender': 6,
      'Balanced Contributor': 7,
      'Support Player': 8,
    },
    topStatNeeds: ['Depth', 'Wing Defense', 'Rebounding'],
    notes: 'Championship contender looking for high-value role players.',
  },
  'Phoenix Mercury': {
    archetypeNeeds: {
      'Primary Creator': 8,
      'Interior Defender': 7,
      'Balanced Contributor': 7,
      'Support Player': 5,
    },
    topStatNeeds: ['Scoring', 'Interior Defense', 'Rebounding'],
    notes: 'Need scoring alongside Brittney Griner and a frontcourt partner.',
  },
  'Seattle Storm': {
    archetypeNeeds: {
      'Primary Creator': 8,
      'Interior Defender': 6,
      'Balanced Contributor': 7,
      'Support Player': 5,
    },
    topStatNeeds: ['Shot Creation', 'Scoring', 'Wing Versatility'],
    notes: 'Transitioning — need a new offensive centerpiece.',
  },
  'Washington Mystics': {
    archetypeNeeds: {
      'Primary Creator': 9,
      'Interior Defender': 8,
      'Balanced Contributor': 9,
      'Support Player': 7,
    },
    topStatNeeds: ['Scoring', 'Defense', 'Rebounding', 'Playmaking'],
    notes: 'Full rebuild — all archetype needs are elevated.',
  },
}
