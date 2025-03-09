# CLAUDE.md - Guide for Working with the Pokémon Agent Codebase

## Overview
This is a framework for building agents that can play Pokémon games using Large Language Models (LLMs).

## Development Commands
- **Setup**: `npm install` (when package.json is created)
- **Run**: `npm start` (when set up)
- **Lint**: `npm run lint` (when set up)
- **Test**: `npm test` (when set up)
- **Test Single File**: `npm test -- -t "test name"` (when set up)

## Code Style Guidelines
- **Formatting**: Use consistent indentation (2 spaces preferred)
- **Naming**: Use camelCase for variables and functions, PascalCase for classes
- **Types**: Use TypeScript types/interfaces for all code
- **Error Handling**: Use try/catch blocks and proper error propagation
- **Comments**: Document complex logic and public APIs
- **Imports**: Group imports by: external libraries, internal modules, types

## Repository Organization
- Organize code by feature/domain related to Pokémon game mechanics
- Keep agent logic separate from game state management