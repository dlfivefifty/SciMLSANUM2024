module SciMLSANUM2024

export compilelab, compilelabsolution, compilelabdemo

#####
# labs
#####

import Literate

function compilelab(k)
    str = replace(read("src/labs/lab$(k)s.jl", String), r"## SOLUTION(.*?)## END"s => "")
    write("labs/lab$k.jl", replace(str, r"## DEMO(.*?)## END"s => s"\1"))
    Literate.notebook("labs/lab$k.jl", "labs/"; execute=false)
end

function compilelabdemo(k)
    str = replace(read("src/labs/lab$(k)s.jl", String), r"## SOLUTION(.*?)## END"s => "")
    write("labs/lab$(k)d.jl", replace(str, r"## DEMO(.*?)## END"s => "##"))
    Literate.notebook("labs/lab$(k)d.jl", "labs/"; execute=false)
end

function compilelabsolution(k)
    Literate.notebook("src/labs/lab$(k)s.jl", "labs/"; execute=false)
end


end # module SciMLSANUM2024
