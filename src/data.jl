"""
    dt2float(dt::DateTime)

Convert a `DateTime` to a decimal year. 
"""
function dt2float(dt::DateTime)
    days = 0
    for m = 1:(month(dt) - 1)
        days += daysinmonth(DateTime(year(dt), month(dt)))
    end
    days += day(dt)
    return year(dt) + days / daysinyear(DateTime(year(dt)))
end

"""
    load_pjm_financials()

Load the inital PJM financials data.
"""
function load_pjm_financials()
    data = JLSO.load("data/pjm_financials.jlso")

    gpf = data[:daily][:GPF][:GPF_6wr10lp11]
    # GPF has the first eleven days missing, so cut those.
    pn = data[:daily][:PN][:PN_def][12:end, :]
    ef = data[:daily][:EF][:EF_6wd07][12:end, :]
    
    # Check that the times are indeed synced.
    @assert pn.time == gpf.time
    @assert ef.time == gpf.time

    times2floats(x) = [dt2float(y.anchor.utc_datetime) for y in x.time]

    return Dict(
        "GPF" => (times2floats(gpf), gpf),
        "PN" => (times2floats(pn), pn),
        "EF" => (times2floats(ef), ef)
    )
end

"""
    load_pjm_ef_nl_financials()

Load financials for EF on PJM with the new node logic.
"""
function load_pjm_ef_nl_financials()
    df = CSV.File("data/PJM_EF_new_logic_financials.jlso.csv") |> DataFrame
    df.date = map(x -> Date(x[5:14], dateformat"y-m-d"), df.time)
    df.time = map(x -> dt2float(DateTime(x)), df.date)
    return df
end

"""
    load_pjm_ef_nl_financials()

Load financials for EF on MISO.
"""
function load_miso_ef_financials()
    df = CSV.File("data/MISO_EF_financials.jlso.csv") |> DataFrame
    df.date = map(x -> Date(x[5:14], dateformat"y-m-d"), df.time)
    df.time = map(x -> dt2float(DateTime(x)), df.date)
    return df
end

"""
    load_temp()
    
Load temperature data.
"""
function load_temp()
    return Dict(
        "MISO" => _load_temp_csv("data/min-mean-max-MISO.21h00.csv"),
        "PJM" => _load_temp_csv("data/min-mean-max-PJM.21h00.csv"),
        "ERCOT" => _load_temp_csv("data/min-mean-max-ERCOT.21h00.csv")
    )
end

function _load_temp_csv(path)
    df = CSV.File(path) |> DataFrame
    df.time = map(dt2float, df.DateTime)
    df.date = map(Date, df.DateTime)
    select!(df, Not("DateTime"))
    return df
end