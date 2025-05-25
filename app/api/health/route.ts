import { NextResponse } from "next/server"

export async function GET() {
  try {
    // Try to reach the backend health endpoint
    const backendHealth = await fetch("http://localhost:8000/health", {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      cache: "no-store"
    })
    
    const isBackendHealthy = backendHealth.ok
    const backendData = isBackendHealthy ? await backendHealth.json() : null
    
    return NextResponse.json({
      status: "ok",
      version: "1.0.0",
      backend: {
        reachable: isBackendHealthy,
        status: backendData?.status || "unreachable",
        version: backendData?.version || null
      },
      timestamp: new Date().toISOString()
    })
  } catch (error) {
    // If we can't reach the backend, report it but don't fail
    return NextResponse.json({
      status: "degraded",
      version: "1.0.0", 
      backend: {
        reachable: false,
        status: "unreachable",
        error: error instanceof Error ? error.message : "Unknown error"
      },
      timestamp: new Date().toISOString()
    })
  }
}
