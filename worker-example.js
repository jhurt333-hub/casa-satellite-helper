const SATELLITE_URL = "https://YOUR-HELPER.example.com/v1/convection";

export async function getCasaConvection(env) {
  const response = await fetch(SATELLITE_URL, {
    headers: { "X-API-Key": env.SATELLITE_HELPER_KEY },
    cf: { cacheTtl: 240, cacheEverything: true },
  });

  if (!response.ok) {
    throw new Error(`Satellite helper failed: ${response.status}`);
  }
  return response.json();
}

// Example inside the Worker's fetch handler:
// if (url.pathname === "/satellite-convection") {
//   return Response.json(await getCasaConvection(env));
// }
