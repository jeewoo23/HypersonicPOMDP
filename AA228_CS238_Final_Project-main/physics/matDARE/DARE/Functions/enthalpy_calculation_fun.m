function enthalpy = enthalpy_calculation_fun(T,specie,Ru,MW_specie)
    % enthalpy_calculation_fun
    %   T        : temperature [K]
    %   specie   : species name string as in janaf_table.txt (e.g. 'O2')
    %   Ru       : universal gas constant [J/(kmol*K)]
    %   MW_specie: molecular weight of species [kg/kmol]
    %
    %   Returns: specific enthalpy [J/kg]

    R = Ru / MW_specie;

    % ---- Cache JANAF species names so we don't re-scan the whole file ----
    persistent Nsh_cached
    if isempty(Nsh_cached)
        fileIDE = fopen('janaf_table.txt','r');
        if fileIDE == -1
            error('enthalpy_calculation_fun:CannotOpenJANAF', ...
                  'Could not open janaf_table.txt');
        end
        Msh = textscan(fileIDE,'%s %s %s %s %s','headerlines',0);
        fclose(fileIDE);
        Nsh_cached = Msh{1,1};  % species name column
    end
    % ----------------------------------------------------------------------

    % ---- Cache numeric coefficients per species as well -------------------
    persistent Dsh_cache
    if isempty(Dsh_cache)
        Dsh_cache = containers.Map('KeyType','char','ValueType','any');
    end

    if isKey(Dsh_cache, specie)
        Dsh = Dsh_cache(specie);
    else
        % Find species block (same logic as original code, but using cached names)
        counter = 1;
        for idx = 1:4:10000
            if idx <= numel(Nsh_cached) && strcmp(Nsh_cached(idx), specie) == 1
                break
            end
            counter = counter + 1;
        end

        if counter == 1 && (numel(Nsh_cached) < 1 || strcmp(Nsh_cached(1),specie) ~= 1)
            error('enthalpy_calculation_fun:SpeciesNotFound', ...
                  'Species "%s" not found in janaf_table.txt', specie);
        end

        m = (counter * 4 - 3);

        fileID = fopen('janaf_table.txt','r');
        if fileID == -1
            error('enthalpy_calculation_fun:CannotOpenJANAF', ...
                  'Could not open janaf_table.txt');
        end
        Csh = textscan(fileID,'%f %f %f %f %f','headerlines',m);
        fclose(fileID);

        % Build Dsh exactly as original enthalpy function (3x5, no special k==5)
        Dsh = zeros(3,5);
        n = 3;
        for k = 1:5
            Ash = Csh{1,k};
            for j = 1:n
                Dsh(j,k) = Ash(j,1);
            end
        end

        % store in cache
        Dsh_cache(specie) = Dsh;
    end
    % ----------------------------------------------------------------------

    % High-T and low-T branches: original enthalpy formula
    if T >= 1000
        enthalpy = R * T * ( ...
            Dsh(1,1)         + ...
            Dsh(1,2)*T/2     + ...
            Dsh(1,3)*(T^2)/3 + ...
            Dsh(1,4)*(T^3)/4 + ...
            Dsh(1,5)*(T^4)/5 + ...
            Dsh(2,1)/T );
    else
        enthalpy = R * T * ( ...
            Dsh(2,3)         + ...
            Dsh(2,4)*T/2     + ...
            Dsh(2,5)*(T^2)/3 + ...
            Dsh(3,1)*(T^3)/4 + ...
            Dsh(3,2)*(T^4)/5 + ...
            Dsh(3,3)/T );
    end
end