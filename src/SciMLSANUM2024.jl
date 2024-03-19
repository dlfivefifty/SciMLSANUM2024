module SciMLSANUM2024

export compilelab, compilelabsolution

#####
# labs
#####

import Literate

function compilelab(k)
    write("labs/lab$k.jl", replace(replace(read("src/labs/lab$(k)s.jl", String), r"## SOLUTION(.*?)## END"s => "")))
    Literate.notebook("labs/lab$k.jl", "labs/"; execute=false)
end

function compilelabsolution(k)
    Literate.notebook("src/labs/lab$(k)s.jl", "labs/"; execute=false)
end


end # module SciMLSANUM2024
