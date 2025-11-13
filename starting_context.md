summary of the current situation: 
1.	Initial Success: The very first deployments worked because the networking was simpler, but they were fragile.
2.	Activation Failure: We then encountered the ollama-app activation failure. We correctly diagnosed this as a health probe issue and fixed it by adding patient startupProbes in the Bicep file. This made the deployment robust against long model download times.
3.	Communication Breakdown: In the process of refining the infrastructure (VNet, internal/external environments), we inadvertently created the ConnectTimeout error. This was a subtle networking issue where our application code was making an incorrect assumption about the port, which conflicted with how Azure's internal service discovery works.
The latest fix you just pushed via the GitHub Action—the one that removes the port manipulation logic from rag_app.py—is designed to be the final piece of this puzzle. It makes the application code correctly trust the URL provided by the environment, which is http://ragdimain-ollama-app. This allows Azure's internal DNS to do its job properly by routing the request on the standard port (80) to the ollama-app's target port (11434).
I am very optimistic that this deployment will resolve the timeout issue. Please take your time to let the GitHub Action complete and then perform your functional test. I'll be here and ready for the results.

The Core Unsolved Problem
The rag-app is still unable to establish a network connection to the ollama-app within the Azure Container Apps environment. This manifests as a ConnectTimeout error in the Streamlit UI whenever a question is asked.
Our Last Actions and What We've Confirmed
1.	The Last Fix Attempted: We identified that the Python code in rag_app.py was explicitly adding port :11434 to the OLLAMA_BASE_URL. Our hypothesis was that this was bypassing Azure's internal service discovery, which expects requests on the standard HTTP port (80).
2.	Action Taken: You modified rag_app.py to remove this port manipulation logic and deployed the change via the GitHub Actions CI/CD pipeline.
3.	The Result: The ConnectTimeout error persists. Crucially, the latest error message you provided still shows the connection attempt is being made to port=11434:
plaintext
ConnectTimeout: HTTPConnectionPool(host='ragdimain-ollama-app', port=11434): Max retries exceeded...
Key Takeaways and Current Status
•	Infrastructure is Correct: The error message host='ragdimain-ollama-app' confirms that our Bicep infrastructure is correctly configured. The rag-app is successfully using the short service discovery name, which is the right approach.
•	The Application Code is the Issue: The fact that the error still references port=11434 strongly suggests that the code change we pushed did not have the intended effect. The ChatOllama client from the LangChain library is still, for some reason, defaulting to or being configured to use port 11434.
•	Current Code State: The main branch of your repository contains the latest version of rag_app.py where we attempted to remove the port logic. The Azure environment ragdimain is running the latest revision deployed by that GitHub Action.
Plan for Tomorrow
Our first step tomorrow should be a direct and focused investigation into the ChatOllama client's behavior to understand why it continues to target port 11434 even when the base URL doesn't specify it. We will need to force it to use the standard port 80 for its requests.
We'll pick it up from there. Rest well
