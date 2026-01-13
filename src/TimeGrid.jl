module TimeGrid

export trapezoid_times_weights

"""
    trapezoid_times_weights(T, Q)

Trapezoidal rule grid on [0,T].
Q = number of subintervals.
Returns:
- times   length Q+1 (0..T)
- weights length Q+1 with endpoints h/2
"""
function trapezoid_times_weights(T::Float64, Q::Int)
    @assert Q >= 1
    h = T / Q
    times = collect(0:h:T)
    weights = fill(h, Q+1)
    weights[1] *= 0.5
    weights[end] *= 0.5
    return times, weights
end

end # module
