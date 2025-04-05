# SmartCred - SaaS Solution "Cloud-based low-cost solution" for Credit Management: BI Aspects
This project involves designing and developing a cloud-based SaaS solution for credit management, specifically targeting decision-makers in small and medium-sized financing companies. The key aspect of this solution is its low cost, making it accessible to smaller financial institutions that may not have the budget for expensive enterprise software.

## Key Components of the Project:
- **Dashboarding & Decision Support System (BI)**
  - The system will include a front-end (FE) and back-end (BE) designed to provide a comprehensive BI (Business Intelligence) dashboard.
  - Decision-makers will be able to visualize credit-related data, analyze trends, and generate reports to support strategic decision-making.
- **Multi-Tenant SaaS Solution**
  -The software will be built as a Software-as-a-Service (SaaS) solution, meaning multiple clients (financing companies) can use the same platform while keeping their data isolated.
  -Multi-tenancy allows cost-sharing, scalability, and easy maintenance.
- **Credit Management Features**
  - The solution will likely cover key aspects of credit management, such as:
    - Loan application processing
    - Credit scoring & risk assessment
    - Payment tracking & delinquency management
    - Customer profiling
    - Reporting & forecasting
- **Efficiency & Configurability**
  - The system will be designed to be efficient, ensuring fast data processing and insights.
  - It will be paramétrable (configurable), meaning clients can customize certain aspects based on their needs.
 
## Links & Configuration requirements:

- https://bun.sh/docs/installation
```
powershell -c "irm bun.sh/install.ps1|iex"
bun --version
bunx create-next-app@latest smart-cred
```
- https://ui.shadcn.com/docs/installation
```
bunx shadcn@latest initcreate-next-app@latest smart-cred
bunx shadcn@latest  init
bun run dev
```





