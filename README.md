# Band Tour Optimization Model

This project is a prototype decision-support system that applies mathematical optimization and data science techniques to the problem of designing a profitable tour for a band. The system models routing, venue selection, and cost management to maximize expected revenue while staying within operational constraints.

The goal is to demonstrate how optimization and analytics can be integrated into the music industry to support strategic decision-making for artists, managers, and promoters.

---

## Key Features

### Tour Routing
- Determines which cities to visit and in what order.
- Formulates the problem as a variant of the Traveling Salesman Problem (TSP).
- Includes feasibility constraints such as maximum travel distance and fuel range limits.

### Budget Management
- Accounts for multiple expense categories, including:
  - Fuel consumption (linked to distance traveled and fuel efficiency).
  - Tolls and parking.
  - Hotel and meal costs.
- Ensures that the overall tour cost does not exceed a defined budget.

### Venue Selection
- Allows each city to have multiple potential venues.
- Selects at most one venue per city.
- Incorporates venue popularity thresholds to ensure only feasible venues are chosen.
- Connects venue selection to city visitation decisions.

### Revenue Optimization
- Objective function maximizes net profit, defined as:
  **Net Profit = Expected Revenue – (Travel + Lodging + Operating Costs)**
- Revenue is based on expected ticket sales and regional fanbase strength.

### Simulation and Analysis
- Displays chosen route, visited cities, and selected venues.
- Provides financial breakdowns for each tour leg (tolls, refueling, lodging, revenue).
- Generates a timeline of the tour schedule.
- Includes map-based visualization of the optimized route.

---

## Technology Stack

- **Python**
- **Pyomo** for optimization modeling.
- **NetworkX** for route handling and graph operations.
- **Matplotlib + Cartopy** for geographic visualization.

---

## Example Workflow

1. **Initialize Data**: Cities, venues, and cost parameters are defined in input data structures.
2. **Build Model**: The optimization model is constructed using Pyomo.
3. **Solve**: The model is solved using a linear/integer programming solver (e.g., CBC).
4. **Analyze**: Results are displayed, including route, venues, days spent per city, and financials.
5. **Visualize**: A U.S. map is generated showing the optimized tour route.

---

## Use Case

This system is intended as a **proof of concept**. It demonstrates how optimization and data-driven modeling can be applied in the music industry to plan tours that balance:
- Audience reach
- Profitability
- Logistical feasibility

While simplified, the framework can be extended with richer data (e.g., dynamic ticket pricing, artist scheduling conflicts, or international routing) to form the basis of a real-world decision-support tool.

---

## Disclaimer

This project is a prototype developed for academic and research purposes. It is not intended for direct production use without further development and validation.
