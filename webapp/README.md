# WNBA Draft Model — Webapp

A web app for displaying WNBA draft needs analysis and player fit rankings.

## Local development

```bash
# 1. Generate player data (from project root)
python prepare_data.py

# 2. Start dev server
cd webapp
npm install
npm run dev
```

Open http://localhost:5173

## Deploying to Vercel (free)

```bash
# Install Vercel CLI once
npm i -g vercel

# From the webapp/ directory:
cd webapp
vercel
```

Follow the prompts. On first deploy, Vercel will ask for a project name and team. Subsequent deploys just run `vercel --prod`.

## Updating player data

When the model is re-run with new data, regenerate `public/players.json`:

```bash
# From project root
python prepare_data.py
```

Then redeploy the webapp.

## Updating team needs

The model outputs archetype need scores per team. Export them as `webapp/public/team_needs.json` with this format:

```json
{
  "Atlanta Dream": {
    "archetypeNeeds": {
      "Primary Creator": 7,
      "Interior Defender": 8,
      "Balanced Contributor": 6,
      "Support Player": 4
    },
    "topStatNeeds": ["Interior Defense", "Rebounding"],
    "notes": "Optional scouting note text."
  }
}
```

The webapp automatically loads this file and uses it instead of the defaults.

## Features

- View all 13 WNBA teams with team needs breakdown
- Per-archetype need scores with visual bars
- Ranked player fits for each team (fit score = need weight × player quality)
- Filter players by archetype and position
- Expand any player row for full stat detail
- Draft tracking — mark players as drafted, they're removed from available fits
- Draft board showing all picked players with undo
- Persisted to localStorage (survives page refresh)
